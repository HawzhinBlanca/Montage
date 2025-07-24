#!/usr/bin/env vault

# HashiCorp Vault Development Configuration for Montage
# This configures a development Vault instance for secret management

storage "inmem" {}

listener "tcp" {
  address = "127.0.0.1:8200"
  tls_disable = true
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true

# Enable development mode
disable_mlock = true
default_lease_ttl = "168h"
max_lease_ttl = "720h"

# Development root token (NEVER use in production)
# For production, use proper authentication methods
log_level = "Info"