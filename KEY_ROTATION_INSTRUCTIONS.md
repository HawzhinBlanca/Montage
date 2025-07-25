# 🚨 EMERGENCY API KEY ROTATION INSTRUCTIONS

**CRITICAL: Do this IMMEDIATELY - your keys are exposed and billing is accumulating**

## Step-by-Step Key Rotation Process

### 1. OpenAI API Key Rotation
```bash
# Old key (REVOKE IMMEDIATELY): sk-proj-tV1gMybmLed_864HZJZmjHMjHQ92X8...
```

**Actions:**
1. Go to https://platform.openai.com/api-keys
2. Find the key starting with `sk-proj-tV1gMybmLed...`
3. Click **Delete** - confirm deletion
4. Create **new key**: Click "Create new secret key"
5. Copy new key to secure location (1Password/similar)
6. **Test old key is dead**: `curl -H "Authorization: Bearer sk-proj-tV1gMybmL..." https://api.openai.com/v1/models`
7. **Should return 401 Unauthorized**

### 2. Anthropic API Key Rotation
```bash
# Old key (REVOKE IMMEDIATELY): sk-ant-api03-4sgNasbi5HxiumJQZe...
```

**Actions:**
1. Go to https://console.anthropic.com/account/keys
2. Find key starting with `sk-ant-api03-4sgNasbi...`
3. Click **Delete** - confirm deletion
4. Create **new key**: Click "Create Key"
5. Copy new key to secure location
6. **Test old key is dead**: `curl -H "x-api-key: sk-ant-api03-4sgNas..." https://api.anthropic.com/v1/messages`
7. **Should return 401/403**

### 3. Deepgram API Key Rotation
```bash
# Old key (REVOKE IMMEDIATELY): ***REMOVED***
```

**Actions:**
1. Go to https://console.deepgram.com/project/keys
2. Find key `27c428c9b064a806...`
3. Click **Delete** - confirm deletion
4. Create **new key**: Click "Create New Key"
5. Copy new key to secure location
6. **Test old key is dead**: `curl -H "Authorization: Token 27c428c9b064a806..." https://api.deepgram.com/v1/projects`
7. **Should return 401**

### 4. Google/Gemini API Key Rotation
```bash
# Old key (REVOKE IMMEDIATELY): ***REMOVED***
```

**Actions:**
1. Go to https://console.cloud.google.com/apis/credentials
2. Find key `AIzaSyBq9lVhGivSuSrK...`
3. Click **Delete** - confirm deletion
4. Create **new key**: Click "Create Credentials" > "API Key"
5. **Restrict the key** to only Gemini API
6. Copy new key to secure location
7. **Test old key is dead**: `curl "https://generativelanguage.googleapis.com/v1/models?key=AIzaSyBq9lVhGivS..."`
8. **Should return 403**

### 5. HuggingFace Token Check
```bash
# Check if this token exists: hf_YOUR_REAL_TOKEN_HERE
```

**Actions:**
1. Go to https://huggingface.co/settings/tokens
2. Check if any tokens exist that shouldn't
3. Delete any suspicious tokens
4. Create new token if needed

## Verification Commands

After rotating all keys, run these tests:

```bash
# Test OLD keys are dead (should all return 401/403)
curl -H "Authorization: Bearer sk-proj-tV1gMybmLed_864HZJZm..." https://api.openai.com/v1/models
curl -H "x-api-key: sk-ant-api03-4sgNasbi5Hxium..." https://api.anthropic.com/v1/messages  
curl -H "Authorization: Token 27c428c9b064a806a622..." https://api.deepgram.com/v1/projects
curl "https://generativelanguage.googleapis.com/v1/models?key=AIzaSyBq9lVhGivSuSrK..."

# Test NEW keys work (should return 200)
curl -H "Authorization: Bearer YOUR_NEW_OPENAI_KEY" https://api.openai.com/v1/models
curl -H "x-api-key: YOUR_NEW_ANTHROPIC_KEY" https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-sonnet-20240229","max_tokens":1,"messages":[{"role":"user","content":"test"}]}'
```

## After Rotation Complete

1. **Update .env with new keys**
2. **Verify .env is in .gitignore** 
3. **Check billing dashboards** for any unexpected charges
4. **Enable billing alerts** on all platforms (set to $10/day max)

## Billing Damage Assessment

Check these billing pages immediately:
- OpenAI: https://platform.openai.com/account/usage
- Anthropic: https://console.anthropic.com/account/billing
- Deepgram: https://console.deepgram.com/billing
- Google Cloud: https://console.cloud.google.com/billing

**If you see unexpected charges, contact support immediately for dispute.**

## Completion Checklist

- [ ] OpenAI key revoked and new one created
- [ ] Anthropic key revoked and new one created  
- [ ] Deepgram key revoked and new one created
- [ ] Google/Gemini key revoked and new one created
- [ ] Old keys return 401/403 errors
- [ ] New keys return 200 status
- [ ] .env file updated with new keys
- [ ] Billing alerts enabled on all platforms
- [ ] Unexpected billing charges disputed if any

**When complete, you can proceed to P0-02 (git history purge)**