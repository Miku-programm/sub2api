package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func openCodeFreeTestAccount() *Account {
	return &Account{ID: 2, Platform: PlatformOpenCodeGo, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"account_mode": "zen", "base_url": "https://opencode.ai/zen/v1"}}
}

// Prefixes independently evaluated with the official v1.18.31 id.ts BigInt algorithm.
func TestOpenCodeIdentifierOfficialSequence(t *testing.T) {
	g := &openCodeIdentifierGenerator{}
	for _, tc := range []struct {
		ms           int64
		desc         bool
		prefix, want string
	}{
		{1792080000000, true, "ses", "ebfb54bffffe"},
		{1792080000000, false, "msg", "1404ab400002"},
		{1792080000000, true, "ses", "ebfb54bffffc"},
		{1792080000001, true, "ses", "ebfb54bfeffe"},
		{1792080000000, false, "msg", "1404ab400001"},
	} {
		id := g.next(tc.prefix, tc.desc, tc.ms)
		require.Equal(t, tc.prefix+"_"+tc.want, id[:16])
		require.Regexp(t, `^[0-9A-Za-z]{14}$`, id[16:])
	}
}

func TestOpenCodeFreeSessionReusedWithinRequest(t *testing.T) {
	c := newOpenCodeSessionTestContext(t, "")
	account := openCodeFreeTestAccount()
	body := []byte(`{"model":"muse-spark-1.3-contributor-free"}`)
	first, retry := make(http.Header), make(http.Header)
	applyOpenCodeSessionHeader(c, account, "https://opencode.ai/zen/v1/responses", first, body)
	applyOpenCodeSessionHeader(c, account, "https://opencode.ai/zen/v1/responses", retry, body)
	require.Equal(t, first.Get(openCodeSessionHeader), retry.Get(openCodeSessionHeader))
	require.Equal(t, first.Get("X-OpenCode-Request"), retry.Get("X-OpenCode-Request"))
	fresh := make(http.Header)
	applyOpenCodeSessionHeader(newOpenCodeSessionTestContext(t, ""), account, "https://opencode.ai/zen/v1/responses", fresh, body)
	require.NotEqual(t, first.Get(openCodeSessionHeader), fresh.Get(openCodeSessionHeader))
}

func TestOpenCodeFreeSessionMapsHintsWithTenantIsolation(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4.1-flash-free","prompt_cache_key":"same-conversation"}`)
	resolve := func(keyID, accountID int64, token string) string {
		c := newOpenCodeSessionTestContext(t, "")
		if keyID != 0 {
			c.Set("api_key", &APIKey{ID: keyID})
		}
		if token != "" {
			c.Request.Header.Set("Authorization", "Bearer "+token)
		}
		a := openCodeFreeTestAccount()
		a.ID = accountID
		h := make(http.Header)
		applyOpenCodeSessionHeader(c, a, "https://opencode.ai/zen/v1/chat/completions", h, body)
		return h.Get(openCodeSessionHeader)
	}
	first := resolve(101, 2, "")
	require.Regexp(t, `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`, first)
	require.Equal(t, first, resolve(101, 2, ""))
	require.NotEqual(t, first, resolve(102, 2, ""))
	require.NotEqual(t, first, resolve(101, 3, ""))
	fallback := resolve(0, 2, "test-only-key-A")
	require.Equal(t, fallback, resolve(0, 2, "test-only-key-A"))
	require.NotEqual(t, fallback, resolve(0, 2, "test-only-key-B"))
	require.NotEqual(t, resolve(0, 2, ""), resolve(0, 2, ""), "anonymous hints must not coalesce across requests")
}

func TestOpenCodeFreeSessionCacheBounded(t *testing.T) {
	cache := &openCodeSessionCache{}
	first := cache.resolve([32]byte{})
	for i := 1; i <= 4096; i++ {
		key := [32]byte{byte(i), byte(i >> 8)}
		cache.resolve(key)
	}
	require.LessOrEqual(t, len(cache.entries), 4096)
	require.NotEqual(t, first, cache.resolve([32]byte{}), "oldest entry must be evicted")
}

func TestOpenCodeFreeSessionCacheExpires(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cache := &openCodeSessionCache{}
		key := [32]byte{1}
		first := cache.resolve(key)
		time.Sleep(23 * time.Hour)
		require.Equal(t, first, cache.resolve(key))
		time.Sleep(25 * time.Hour)
		require.NotEqual(t, first, cache.resolve(key))
	})
}

func TestOpenCodeFreeHeadersAndCallerPayloadPreserved(t *testing.T) {
	for _, tc := range []struct{ name, body, ua, path string }{
		{"responses", `{"model":"muse-spark-1.3-contributor-free", "store":false,"stream":true,"instructions":"CUSTOM SYSTEM", "input":[{"role":"user","content":"first"},{"role":"assistant","content":"answer"},{"type":"function_call","call_id":"call1","name":"lookup_test_value","arguments":"{}"},{"type":"function_call_output","call_id":"call1","output":"custom tool result"}],"tools":[{"type":"function","name":"lookup_test_value","parameters":{"type":"object","properties":{"key":{"type":"string"}}}}],"tool_choice":"auto"}`, "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14", "/responses"},
		{"chat", `{"model":"deepseek-v4.1-flash-free", "stream":false,"messages":[{"role":"system","content":"CUSTOM SYSTEM"},{"role":"user","content":"first"},{"role":"assistant","reasoning_content":"kept reasoning","content":null,"tool_calls":[{"id":"call1","type":"function","function":{"name":"lookup_test_value","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call1","content":"custom tool result"}],"tools":[{"type":"function","function":{"name":"lookup_test_value","parameters":{"type":"object","properties":{"key":{"type":"string"}}}}}],"tool_choice":"none"}`, "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14", "/chat/completions"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := openCodeSessionTestService()
			c := newOpenCodeSessionTestContext(t, "caller-arbitrary-session")
			c.Set("api_key", &APIKey{ID: 201})
			c.Request.Header.Set("User-Agent", "generic-caller/1")
			body := []byte(tc.body)
			var req *http.Request
			if tc.name == "responses" {
				var err error
				req, err = svc.buildUpstreamRequest(context.Background(), c, openCodeFreeTestAccount(), body, "test-upstream-token", false, "", false)
				require.NoError(t, err)
			} else {
				upstream := &openCodeSessionHTTPUpstream{}
				svc.httpUpstream = upstream
				resp, err := svc.sendCCUpstreamRequest(context.Background(), c, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1"+tc.path, body, false, "test-upstream-token", "", "")
				require.NoError(t, err)
				require.NoError(t, resp.Body.Close())
				req = upstream.request
			}
			require.Equal(t, tc.ua, req.Header.Get("User-Agent"))
			require.Equal(t, "cli", req.Header.Get("X-OpenCode-Client"))
			require.Equal(t, "global", req.Header.Get("X-OpenCode-Project"))
			require.Regexp(t, `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`, req.Header.Get(openCodeSessionHeader))
			require.Regexp(t, `^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$`, req.Header.Get("X-OpenCode-Request"))
			require.Equal(t, "https://opencode.ai/zen/v1"+tc.path, req.URL.String())
			out, err := io.ReadAll(req.Body)
			require.NoError(t, err)
			require.Equal(t, tc.body, string(out), "system, history, tools, choices and whitespace must be unchanged")
			require.Equal(t, tc.body, string(body))
		})
	}
}

func TestOpenCodeFreeNativeIdentityAndOverrides(t *testing.T) {
	account := openCodeFreeTestAccount()
	account.Credentials[credKeyHeaderOverrideEnabled] = true
	account.Credentials[credKeyHeaderOverrides] = map[string]any{"user-agent": "explicit-agent", "x-opencode-client": "explicit-client", "x-opencode-project": "explicit-project"}
	session := "ses_ebfb54bffffe0123456789ABCD"
	request := "msg_1404ab4000020123456789ABCD"
	c := newOpenCodeSessionTestContext(t, session)
	c.Request.Header.Set("X-OpenCode-Request", request)
	body := []byte(`{"model":"muse-spark-1.3-contributor-free"}`)
	h := make(http.Header)
	account.ApplyHeaderOverrides(h)
	applyOpenCodeSessionHeader(c, account, "https://opencode.ai/zen/v1/responses", h, body)
	require.Equal(t, session, h.Get(openCodeSessionHeader))
	require.Equal(t, request, h.Get("X-OpenCode-Request"))
	require.Equal(t, "explicit-agent", h.Get("User-Agent"))
	require.Equal(t, "explicit-client", h.Get("X-OpenCode-Client"))
	require.Equal(t, "explicit-project", h.Get("X-OpenCode-Project"))
}

func TestOpenCodeFreeIdentityScope(t *testing.T) {
	for _, tc := range []struct {
		name, platform, accountType, mode, url, model string
		want                                          bool
	}{
		{"muse zen", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", true},
		{"deepseek inference", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/inference/v1/chat/completions", "deepseek-v4.1-flash-free", true},
		{"inference openai", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://OPENCODE.AI:443/inference/openai/v1/responses", "muse-spark-1.3-contributor-free", true},
		{"paid", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3", false},
		{"other free", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/chat/completions", "big-pickle", false},
		{"typo", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributior-free", false},
		{"alias", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "opencode/muse-spark-1.3-contributor-free", false},
		{"model suffix", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free-extra", false},
		{"other provider", PlatformOpenAI, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"oauth", PlatformOpenCodeGo, AccountTypeOAuth, "zen", "https://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"go plan", PlatformOpenCodeGo, AccountTypeAPIKey, "go", "https://opencode.ai/zen/go/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"custom relay", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://relay.example/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"lookalike", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai.evil.example/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"subdomain", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://api.opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"http", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "http://opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"port", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai:8443/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"userinfo", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://user@opencode.ai/zen/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"unrelated path", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/other/responses", "muse-spark-1.3-contributor-free", false},
		{"prefix lookalike", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zenish/v1/responses", "muse-spark-1.3-contributor-free", false},
		{"path traversal", PlatformOpenCodeGo, AccountTypeAPIKey, "zen", "https://opencode.ai/zen/../other/responses", "muse-spark-1.3-contributor-free", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := openCodeFreeTestAccount()
			a.Platform, a.Type = tc.platform, tc.accountType
			a.Credentials["account_mode"] = tc.mode
			body := []byte(`{"model":` + strconv.Quote(tc.model) + `}`)
			c := newOpenCodeSessionTestContext(t, "not-native-session")
			// The outbound mapped body, not this remembered body, determines scope.
			rememberOpenCodeInboundBody(c, []byte(`{"model":"muse-spark-1.3-contributor-free"}`))
			h := http.Header{"User-Agent": {"sentinel-agent"}}
			applyOpenCodeSessionHeader(c, a, tc.url, h, body)
			if tc.want {
				require.NotEmpty(t, h.Get("X-OpenCode-Request"))
				require.Equal(t, "cli", h.Get("X-OpenCode-Client"))
			} else {
				require.Empty(t, h.Get("X-OpenCode-Request"))
				require.Empty(t, h.Get("X-OpenCode-Client"))
				require.Equal(t, "sentinel-agent", h.Get("User-Agent"))
				if got := h.Get(openCodeSessionHeader); got != "" {
					require.Equal(t, "not-native-session", got)
				}
			}
			require.Equal(t, `{"model":`+strconv.Quote(tc.model)+`}`, string(body))
		})
	}
}

func TestOpenCodeIdentifierConcurrency(t *testing.T) {
	g := &openCodeIdentifierGenerator{}
	var wg sync.WaitGroup
	ids := make(chan string, 1024)
	for i := 0; i < 1024; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); ids <- g.next("msg", false, 1792080000000) }()
	}
	wg.Wait()
	close(ids)
	counters := make(map[uint64]bool)
	for id := range ids {
		n, err := strconv.ParseUint(id[4:16], 16, 64)
		require.NoError(t, err)
		counter := n & 0xfff
		require.False(t, counters[counter], "counter must be atomic even with identical milliseconds")
		counters[counter] = true
	}
	require.Len(t, counters, 1024)
	require.True(t, counters[1])
	require.True(t, counters[1024])
}

func TestOpenCodeFreeSessionConcurrentRequests(t *testing.T) {
	var wg sync.WaitGroup
	ids := make(chan string, 64)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			c.Request.Header.Set(openCodeSessionHeader, "concurrent-conversation")
			c.Set("api_key", &APIKey{ID: 301})
			h := make(http.Header)
			applyOpenCodeSessionHeader(c, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/responses", h, []byte(`{"model":"muse-spark-1.3-contributor-free"}`))
			ids <- h.Get(openCodeSessionHeader)
		}()
	}
	wg.Wait()
	close(ids)
	first := ""
	for id := range ids {
		if first == "" {
			first = id
		}
		require.Equal(t, first, id)
	}
}

func TestOpenCodeFreeNativeValidationAndHintPrecedence(t *testing.T) {
	for _, hint := range []string{"ses_bad", "ses_ebfb54bffffe0123456789ABC!", "msg_1404ab4000020123456789ABCD", "ses_EBFB54BFFFFE0123456789ABCD", "ses_ebfb54bffffe0123456789ABCDextra", "bad\r\nhint"} {
		c := newOpenCodeSessionTestContext(t, hint)
		c.Request.Header.Set("X-OpenCode-Request", hint)
		h := make(http.Header)
		applyOpenCodeSessionHeader(c, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/responses", h, []byte(`{"model":"muse-spark-1.3-contributor-free"}`))
		require.Regexp(t, `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`, h.Get(openCodeSessionHeader))
		if !strings.HasPrefix(hint, "msg_") {
			require.NotEqual(t, hint, h.Get("X-OpenCode-Request"))
		}
	}
	c := newOpenCodeSessionTestContext(t, "header-hint")
	c.Set("api_key", &APIKey{ID: 401})
	first := make(http.Header)
	applyOpenCodeSessionHeader(c, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/responses", first, []byte(`{"model":"muse-spark-1.3-contributor-free","prompt_cache_key":"body-hint"}`))
	c2 := newOpenCodeSessionTestContext(t, "")
	c2.Set("api_key", &APIKey{ID: 401})
	rememberOpenCodeInboundBody(c2, []byte(`{"model":"caller-alias","metadata":{"user_id":"{\"session_id\":\"header-hint\"}"}}`))
	second := make(http.Header)
	applyOpenCodeSessionHeader(c2, openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/chat/completions", second, []byte(`{"model":"deepseek-v4.1-flash-free"}`))
	require.Equal(t, first.Get(openCodeSessionHeader), second.Get(openCodeSessionHeader), "preserve session across turns, conversion and model changes")
	require.NotEqual(t, first.Get("X-OpenCode-Request"), second.Get("X-OpenCode-Request"))
}

func TestOpenCodeFreeIdentityGeneratedInNativeOrder(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := make(http.Header)
		applyOpenCodeSessionHeader(newOpenCodeSessionTestContext(t, ""), openCodeFreeTestAccount(), "https://opencode.ai/zen/v1/responses", h, []byte(`{"model":"muse-spark-1.3-contributor-free"}`))
		ses, err := strconv.ParseUint(h.Get(openCodeSessionHeader)[4:16], 16, 64)
		require.NoError(t, err)
		msg, err := strconv.ParseUint(h.Get("X-OpenCode-Request")[4:16], 16, 64)
		require.NoError(t, err)
		require.Equal(t, ((^ses&0xffffffffffff)+1)&0xffffffffffff, msg, "session descending ID is created before ascending request ID")
	})
}

func TestOpenCodeFreeExplicitSessionAndRequestOverrides(t *testing.T) {
	a := openCodeFreeTestAccount()
	a.Credentials[credKeyHeaderOverrideEnabled] = true
	a.Credentials[credKeyHeaderOverrides] = map[string]any{"x-opencode-session": "explicit-session", "x-opencode-request": "explicit-request"}
	body := []byte(`{"model":"muse-spark-1.3-contributor-free"}`)
	for _, hint := range []string{"", "ses_ebfb54bffffe0123456789ABCD"} {
		h := make(http.Header)
		a.ApplyHeaderOverrides(h)
		applyOpenCodeSessionHeader(newOpenCodeSessionTestContext(t, hint), a, "https://opencode.ai/zen/v1/responses", h, body)
		wantSession := hint
		if hint == "" {
			wantSession = "explicit-session"
		}
		require.Equal(t, wantSession, h.Get(openCodeSessionHeader), "retain existing caller-over-account session precedence")
		require.Equal(t, "explicit-request", h.Get("X-OpenCode-Request"))
	}
}

func TestOpenCodeFreePassthroughBuilderPreservesBody(t *testing.T) {
	svc := openCodeSessionTestService()
	// The existing Responses builder defaults store=false; supplying it here
	// isolates byte-preservation of caller system/history/tools from that policy.
	body := []byte(`{ "model":"muse-spark-1.3-contributor-free", "store":false, "stream":false, "input":[{"role":"system","content":"caller-system"},{"role":"user","content":"hello"}], "tools":[], "metadata":{"kept":"yes"} }`)
	c := newOpenCodeSessionTestContext(t, "")
	req, err := svc.buildUpstreamRequestOpenAIPassthrough(context.Background(), c, openCodeFreeTestAccount(), body, "test-upstream-token")
	require.NoError(t, err)
	out, err := io.ReadAll(req.Body)
	require.NoError(t, err)
	require.Equal(t, body, out)
	require.Equal(t, "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14", req.Header.Get("User-Agent"))
	require.Equal(t, "https://opencode.ai/zen/v1/responses", req.URL.String())
}

func TestOpenCodeFreeSessionGenerated(t *testing.T) {
	headers := make(http.Header)
	applyOpenCodeSessionHeader(newOpenCodeSessionTestContext(t, ""), openCodeFreeTestAccount(),
		"https://opencode.ai/zen/v1/responses", headers,
		[]byte(`{"model":"muse-spark-1.3-contributor-free","input":"hi"}`))
	require.Regexp(t, `^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`, headers.Get(openCodeSessionHeader))
}
