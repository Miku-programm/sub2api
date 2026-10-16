package service

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenCodeScopedIdentityTrustBoundary(t *testing.T) {
	const muse = "muse-spark-1.3-contributor-free"
	for _, tc := range []struct {
		name, target, model, platform, mode, kind string
		want                                      bool
	}{
		{"zen", "https://opencode.ai/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, true},
		{"inference", "https://opencode.ai/inference/v1/chat/completions", "deepseek-v4.1-flash-free", PlatformOpenCodeGo, "zen", AccountTypeAPIKey, true},
		{"upper host default port", "https://OPENCODE.AI:443/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, true},
		{"lookalike", "https://opencode.ai.evil.example/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"subdomain", "https://api.opencode.ai/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"http", "http://opencode.ai/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"userinfo", "https://user@opencode.ai/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"port", "https://opencode.ai:8443/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"other path", "https://opencode.ai/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"prefix path", "https://opencode.ai/zenith/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"paid Muse", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3", PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"paid DeepSeek", "https://opencode.ai/zen/v1/chat/completions", "deepseek-v4.1-flash", PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"alias not mapped", "https://opencode.ai/zen/v1/responses", "opencode/" + muse, PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"typo literal", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributior-free", PlatformOpenCodeGo, "zen", AccountTypeAPIKey, false},
		{"other provider", "https://opencode.ai/zen/v1/responses", muse, PlatformOpenAI, "zen", AccountTypeAPIKey, false},
		{"Go unchanged", "https://opencode.ai/zen/go/v1/responses", muse, PlatformOpenCodeGo, "go", AccountTypeAPIKey, false},
		{"OAuth unchanged", "https://opencode.ai/zen/v1/responses", muse, PlatformOpenCodeGo, "zen", AccountTypeOAuth, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := &Account{ID: 2, Platform: tc.platform, Type: tc.kind, Credentials: map[string]any{"account_mode": tc.mode}}
			body := []byte(fmt.Sprintf(`{"model":%q}`, tc.model))
			c := newOpenCodeSessionTestContext(t, "")
			// Inbound free-model hints must not broaden eligibility after mapping.
			rememberOpenCodeInboundBody(c, []byte(`{"model":"muse-spark-1.3-contributor-free"}`))
			h := http.Header{"User-Agent": {"sentinel-agent"}}
			applyOpenCodeSessionHeader(c, a, tc.target, h, body)
			require.Equal(t, tc.want, h.Get("X-OpenCode-Client") == "cli")
			if !tc.want {
				require.Equal(t, "sentinel-agent", h.Get("User-Agent"))
				require.Empty(t, h.Get("X-OpenCode-Request"))
				require.Empty(t, h.Get("X-OpenCode-Project"))
			}
		})
	}
}

func TestOpenCodeScopedIdentityConcurrentGeneration(t *testing.T) {
	var g openCodeIdentifierGenerator
	const n = 1024
	ids := make(chan string, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); ids <- g.next("msg", false, 1792080000000) }()
	}
	wg.Wait()
	close(ids)
	prefixes := make(map[string]bool)
	for id := range ids {
		require.Regexp(t, `^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$`, id)
		require.False(t, prefixes[id[:16]], "counter must be unique even before random suffix")
		prefixes[id[:16]] = true
	}
	require.Len(t, prefixes, n)
}

func TestOpenCodeScopedIdentityConcurrentMapping(t *testing.T) {
	var cache openCodeSessionCache
	ids := make(chan string, 128)
	var wg sync.WaitGroup
	for i := 0; i < cap(ids); i++ {
		wg.Add(1)
		go func() { defer wg.Done(); ids <- cache.resolve([32]byte{7}) }()
	}
	wg.Wait()
	close(ids)
	first := <-ids
	for id := range ids {
		require.Equal(t, first, id)
	}
}

func TestOpenCodeScopedIdentityDoesNotTouchBodiesOrDuplicateHeaders(t *testing.T) {
	for _, model := range []string{"muse-spark-1.3-contributor-free", "deepseek-v4.1-flash-free"} {
		body := []byte(fmt.Sprintf(`{ "model":%q, "messages":[{"role":"system","content":"custom"},{"role":"user","content":"hi"}] }`, model))
		before := string(body)
		c := newOpenCodeSessionTestContext(t, "invalid-native-session")
		c.Request.Header.Set("X-OpenCode-Request", "msg_bad\nrequest")
		h := http.Header{
			"user-agent": {"old-one"}, "USER-AGENT": {"old-two"},
			"x-opencode-client": {"old-client"}, "X-OPENCODE-CLIENT": {"old-client-two"},
			"x-opencode-session": {"old-session"}, "X-OPENCODE-SESSION": {"old-session-two"},
			"x-opencode-project": {"old-project"}, "X-OPENCODE-PROJECT": {"old-project-two"},
			"x-opencode-request": {"old-request"}, "X-OPENCODE-REQUEST": {"old-request-two"},
		}
		applyOpenCodeSessionHeader(c, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/responses", h, body)
		require.Equal(t, before, string(body), "must not invent tools/system or rewrite no-tools calls")
		for _, name := range []string{"User-Agent", "X-OpenCode-Client", "X-OpenCode-Session", "X-OpenCode-Project", "X-OpenCode-Request"} {
			count := 0
			for key, values := range h {
				if strings.EqualFold(name, key) {
					count += len(values)
				}
			}
			require.Equal(t, 1, count, name)
		}
		require.Regexp(t, `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`, h.Get(openCodeSessionHeader))
		require.Regexp(t, `^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$`, h.Get("X-OpenCode-Request"))
	}
}
