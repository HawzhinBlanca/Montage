#!/bin/bash
# Vault setup script for Montage secrets management

set -e

VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_ADDR

echo "🔐 Setting up Vault for Montage..."

# Check if vault is running
if ! vault status >/dev/null 2>&1; then
    echo "❌ Vault is not running. Please start vault server first:"
    echo "   vault server -dev -config=infrastructure/vault/dev-config.hcl"
    exit 1
fi

echo "✅ Vault is running at $VAULT_ADDR"

# Enable KV v2 secrets engine at secret/
echo "📝 Enabling KV v2 secrets engine..."
vault secrets enable -path=secret kv-v2 || echo "KV engine already enabled"

# Create Montage secrets
echo "🔑 Creating Montage API secrets..."

# Note: In production, these would be real rotated keys
vault kv put secret/montage/openai \
    OPENAI_API_KEY="sk-proj-YOUR_NEW_OPENAI_KEY_AFTER_ROTATION"

vault kv put secret/montage/anthropic \
    ANTHROPIC_API_KEY="sk-ant-YOUR_NEW_ANTHROPIC_KEY_AFTER_ROTATION"

vault kv put secret/montage/deepgram \
    DEEPGRAM_API_KEY="YOUR_NEW_DEEPGRAM_KEY_AFTER_ROTATION"

vault kv put secret/montage/gemini \
    GEMINI_API_KEY="YOUR_NEW_GEMINI_KEY_AFTER_ROTATION"

echo "✅ Vault secrets created successfully!"

# Create policy for Montage application
echo "📋 Creating Montage access policy..."
vault policy write montage-policy - <<EOF
# Montage application policy - read-only access to API keys
path "secret/data/montage/*" {
  capabilities = ["read"]
}

path "secret/metadata/montage/*" {
  capabilities = ["list", "read"]
}
EOF

echo "✅ Vault setup complete!"
echo ""
echo "🔍 Testing secret access..."
vault kv get secret/montage/openai
echo ""
echo "🎯 Run 'make vault-smoke' to verify integration"