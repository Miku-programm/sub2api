package service

import (
	"container/list"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

var openCodeNativeSessionPattern = regexp.MustCompile(`^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`)
var openCodeNativeRequestPattern = regexp.MustCompile(`^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$`)

// Identity only: do not synthesize prompts/tools or rewrite any request body.
// These defaults are scoped by openCodeFreeIdentityModel at the shared hook.
func applyOpenCodeFreeHeaders(c *gin.Context, account *Account, headers http.Header, model string) {
	version := "4.0.23"
	if model == "muse-spark-1.3-contributor-free" {
		version = "4.0.40"
	}
	requestID := ""
	if c != nil && c.Request != nil {
		if id := c.GetHeader("X-OpenCode-Request"); openCodeNativeRequestPattern.MatchString(id) {
			requestID = id
		}
	}
	key := fmt.Sprintf("opencode_free_request_%d", account.ID)
	if requestID == "" && c != nil {
		requestID = c.GetString(key)
	}
	if requestID == "" {
		// The session is always resolved first, just as in the native client.
		requestID = openCodeIdentifiers.next("msg", false, time.Now().UnixMilli())
		if c != nil {
			c.Set(key, requestID)
		}
	}
	defaults := map[string]string{
		"User-Agent":         "opencode/1.18.31 ai-sdk/provider-utils/" + version + " runtime/bun/1.3.14",
		"X-OpenCode-Client":  "cli",
		"X-OpenCode-Project": "global",
		"X-OpenCode-Request": requestID,
	}
	for name, value := range defaults {
		if override, ok := account.HeaderOverrideValue(strings.ToLower(name)); ok {
			value = override
		}
		for key := range headers {
			if strings.EqualFold(key, name) {
				delete(headers, key)
			}
		}
		headers.Set(name, value)
	}
}

// Cache keys contain only a SHA-256 digest, never plaintext authorization or hints.
// The model is deliberately excluded so a conversation can switch free models.
// A process-local, 4096-entry LRU with a 24-hour idle TTL. Eviction or restart
// starts a new upstream identity; it never stores or modifies conversation data.
type openCodeSessionCache struct {
	mu      sync.Mutex
	entries map[[32]byte]*list.Element
	lru     list.List
}

type openCodeSessionCacheEntry struct {
	key     [32]byte
	id      string
	expires time.Time
}

var openCodeFreeSessions openCodeSessionCache

func openCodeSessionEntry(element *list.Element) *openCodeSessionCacheEntry {
	entry, ok := element.Value.(*openCodeSessionCacheEntry)
	if !ok {
		panic("invalid OpenCode session cache entry")
	}
	return entry
}

func (cache *openCodeSessionCache) resolve(key [32]byte) string {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	now := time.Now()
	if cache.entries == nil {
		cache.entries = make(map[[32]byte]*list.Element)
	}
	for old := cache.lru.Back(); old != nil; old = cache.lru.Back() {
		entry := openCodeSessionEntry(old)
		if now.Before(entry.expires) {
			break
		}
		delete(cache.entries, entry.key)
		cache.lru.Remove(old)
	}
	if element := cache.entries[key]; element != nil {
		entry := openCodeSessionEntry(element)
		entry.expires = now.Add(24 * time.Hour)
		cache.lru.MoveToFront(element)
		return entry.id
	}
	if cache.lru.Len() >= 4096 {
		old := cache.lru.Back()
		delete(cache.entries, openCodeSessionEntry(old).key)
		cache.lru.Remove(old)
	}
	id := newOpenCodeNativeSession()
	cache.entries[key] = cache.lru.PushFront(&openCodeSessionCacheEntry{key: key, id: id, expires: now.Add(24 * time.Hour)})
	return id
}

func resolveOpenCodeFreeSession(c *gin.Context, account *Account, headers http.Header, bodies ...[]byte) string {
	// Preserve the existing caller-header/body precedence over account fallbacks.
	hint := resolveOpenCodeSessionID(c, nil, false, bodies...)
	if openCodeNativeSessionPattern.MatchString(hint) {
		return hint
	}
	if hint == "" {
		if override, ok := account.HeaderOverrideValue("x-opencode-session"); ok {
			return override
		}
		if id := existingOpenCodeSessionHeader(headers); openCodeNativeSessionPattern.MatchString(id) {
			return id
		}
	}
	apiKeyID := getAPIKeyIDFromContext(c)
	tenant := fmt.Sprintf("key:%d", apiKeyID)
	identified := apiKeyID > 0
	if !identified && c != nil && c.Request != nil {
		if auth := c.GetHeader("Authorization"); auth != "" {
			tenant = fmt.Sprintf("authorization:%x", sha256.Sum256([]byte(auth)))
			identified = true
		} else if auth := c.GetHeader("X-Api-Key"); auth != "" {
			tenant = fmt.Sprintf("x-api-key:%x", sha256.Sum256([]byte(auth)))
			identified = true
		}
	}
	key := sha256.Sum256([]byte(fmt.Sprintf("%d:%s:%s", account.ID, tenant, hint)))
	contextKey := fmt.Sprintf("opencode_free_session_%x", key)
	if c != nil {
		if id := c.GetString(contextKey); id != "" {
			return id
		}
	}
	var id string
	if hint != "" && identified {
		id = openCodeFreeSessions.resolve(key)
	} else {
		id = newOpenCodeNativeSession()
	}
	if c != nil {
		c.Set(contextKey, id)
	}
	return id
}

// Only mapped free model IDs on the official Zen/inference API opt in.
// Never infer eligibility from a caller alias or a remembered inbound body.
func openCodeFreeIdentityModel(account *Account, targetURL string, bodies ...[]byte) string {
	if account == nil || account.Type != AccountTypeAPIKey || !account.IsOpenCodeZen() || len(bodies) == 0 {
		return ""
	}
	u, err := url.Parse(targetURL)
	if err != nil || !strings.EqualFold(u.Scheme, "https") || !strings.EqualFold(u.Hostname(), "opencode.ai") ||
		u.User != nil || (u.Port() != "" && u.Port() != "443") || u.Path != path.Clean(u.Path) ||
		(!strings.HasPrefix(u.Path, "/zen/") && !strings.HasPrefix(u.Path, "/inference/")) {
		return ""
	}
	model := gjson.GetBytes(bodies[0], "model").String()
	switch model {
	case "muse-spark-1.3-contributor-free", "deepseek-v4.1-flash-free":
		return model
	default:
		return ""
	}
}

type openCodeIdentifierGenerator struct {
	mu            sync.Mutex
	lastTimestamp int64
	counter       uint64
}

var openCodeIdentifiers openCodeIdentifierGenerator

// Matches OpenCode v1.18.31 packages/opencode/src/id/id.ts: shared counter,
// reset on any millisecond change, increment before encoding, invert descending
// IDs, keep the low six bytes, then append 14 random base62 characters.
func (g *openCodeIdentifierGenerator) next(prefix string, descending bool, timestamp int64) string {
	g.mu.Lock()
	defer g.mu.Unlock()
	if timestamp != g.lastTimestamp {
		g.lastTimestamp = timestamp
		g.counter = 0
	}
	g.counter++
	n := uint64(timestamp)*0x1000 + g.counter
	if descending {
		n = ^n
	}
	const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	var suffix [14]byte
	_, _ = rand.Read(suffix[:])
	for i := range suffix {
		suffix[i] = alphabet[int(suffix[i])%len(alphabet)]
	}
	return fmt.Sprintf("%s_%012x%s", prefix, n&0xffffffffffff, suffix[:])
}

func newOpenCodeNativeSession() string {
	return openCodeIdentifiers.next("ses", true, time.Now().UnixMilli())
}
