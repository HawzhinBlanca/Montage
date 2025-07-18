# AI Video Processing Pipeline - Production Infrastructure
# Terraform configuration for cloud deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "video-pipeline-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "AI-Video-Pipeline"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "video-pipeline"
}

variable "node_group_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "node_group_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "node_group_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # EKS requirements
  public_subnet_tags = {
    "kubernetes.io/role/elb"                        = "1"
    "kubernetes.io/cluster/${var.cluster_name}"     = "shared"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${var.cluster_name}"     = "shared"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id                    = module.vpc.vpc_id
  subnet_ids                = module.vpc.private_subnets
  control_plane_subnet_ids  = module.vpc.private_subnets
  
  # Cluster access
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  
  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      name = "main-node-group"
      
      instance_types = ["c5.2xlarge"]  # 8 vCPU, 16 GiB RAM
      
      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size
      
      # Node group configuration
      ami_type       = "AL2_x86_64"
      capacity_type  = "ON_DEMAND"
      disk_size      = 100
      
      # Taints for video processing workloads
      taints = []
      
      # Labels
      labels = {
        workload = "video-processing"
      }
      
      # Launch template
      create_launch_template = true
      launch_template_tags = {
        Name = "${var.cluster_name}-node-template"
      }
    }
    
    # GPU nodes for acceleration
    gpu = {
      name = "gpu-node-group"
      
      instance_types = ["g4dn.xlarge"]  # 1 GPU, 4 vCPU, 16 GiB RAM
      
      min_size     = 0
      max_size     = 5
      desired_size = 0  # Start with 0, scale up on demand
      
      ami_type      = "AL2_x86_64_GPU"
      capacity_type = "SPOT"  # Use spot instances for cost optimization
      disk_size     = 100
      
      labels = {
        workload = "video-processing-gpu"
        node-type = "gpu"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # aws-auth configmap
  manage_aws_auth_configmap = true
  
  aws_auth_roles = [
    {
      rolearn  = module.eks.eks_managed_node_groups.main.iam_role_arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups   = ["system:bootstrappers", "system:nodes"]
    },
  ]
}

# RDS PostgreSQL
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  
  identifier = "${var.cluster_name}-postgres"
  
  # Engine
  engine               = "postgres"
  engine_version       = "15.4"
  family              = "postgres15"
  major_engine_version = "15"
  instance_class       = var.db_instance_class
  
  # Storage
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  # Database
  db_name  = "video_pipeline"
  username = "video_user"
  port     = 5432
  
  # Networking
  db_subnet_group_name   = module.rds.db_subnet_group_name
  vpc_security_group_ids = [module.security_groups.rds_security_group_id]
  
  # Backup
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  # Monitoring
  monitoring_interval    = 60
  monitoring_role_name   = "${var.cluster_name}-rds-monitoring"
  create_monitoring_role = true
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Enhanced monitoring
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  # Read replicas
  create_read_replica = true
  read_replica_identifier = "${var.cluster_name}-postgres-replica"
  
  tags = {
    Name = "${var.cluster_name}-postgres"
  }
}

# ElastiCache Redis
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  
  # Cluster
  cluster_id         = "${var.cluster_name}-redis"
  description        = "Redis cluster for video pipeline"
  
  # Engine
  engine          = "redis"
  engine_version  = "7.0"
  node_type       = var.redis_node_type
  port            = 6379
  
  # Cluster configuration
  num_cache_clusters      = 3
  replication_group_id    = "${var.cluster_name}-redis"
  parameter_group_name    = "default.redis7"
  
  # Networking
  subnet_group_name = module.redis.elasticache_subnet_group_name
  security_group_ids = [module.security_groups.redis_security_group_id]
  
  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  # Maintenance
  maintenance_window = "sun:05:00-sun:09:00"
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = true
  
  tags = {
    Name = "${var.cluster_name}-redis"
  }
}

# Security Groups
module "security_groups" {
  source = "./modules/security-groups"
  
  vpc_id      = module.vpc.vpc_id
  vpc_cidr    = module.vpc.vpc_cidr_block
  cluster_name = var.cluster_name
}

# S3 Buckets
resource "aws_s3_bucket" "video_storage" {
  bucket = "${var.cluster_name}-video-storage-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_versioning" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id
  
  rule {
    id     = "video_lifecycle"
    status = "Enabled"
    
    # Move to IA after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    # Move to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    # Delete after 365 days
    expiration {
      days = 365
    }
  }
}

# Random string for unique bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"
  
  cluster_name = var.cluster_name
  s3_bucket_arn = aws_s3_bucket.video_storage.arn
}

# Application Load Balancer
module "alb" {
  source = "terraform-aws-modules/alb/aws"
  
  name               = "${var.cluster_name}-alb"
  load_balancer_type = "application"
  
  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets
  
  security_groups = [module.security_groups.alb_security_group_id]
  
  # Access logs
  access_logs = {
    bucket  = aws_s3_bucket.alb_logs.bucket
    enabled = true
  }
  
  target_groups = [
    {
      name             = "${var.cluster_name}-api"
      backend_protocol = "HTTP"
      backend_port     = 80
      target_type      = "ip"
      
      health_check = {
        enabled             = true
        healthy_threshold   = 2
        interval            = 30
        matcher             = "200"
        path                = "/health"
        port                = "traffic-port"
        protocol            = "HTTP"
        timeout             = 5
        unhealthy_threshold = 2
      }
    }
  ]
  
  https_listeners = [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = module.acm.acm_certificate_arn
      target_group_index = 0
    }
  ]
  
  http_tcp_listeners = [
    {
      port        = 80
      protocol    = "HTTP"
      action_type = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  ]
}

# ALB access logs bucket
resource "aws_s3_bucket" "alb_logs" {
  bucket = "${var.cluster_name}-alb-logs-${random_string.bucket_suffix.result}"
}

# ACM Certificate
module "acm" {
  source = "terraform-aws-modules/acm/aws"
  
  domain_name = var.domain_name
  zone_id     = data.aws_route53_zone.main.zone_id
  
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.${var.domain_name}",
  ]
  
  wait_for_validation = true
}

# Route53 Zone (assuming it exists)
data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

# Domain variable
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "video-pipeline.example.com"
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.redis.elasticache_replication_group_primary_endpoint
}

output "s3_bucket" {
  description = "S3 bucket for video storage"
  value       = aws_s3_bucket.video_storage.bucket
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = module.alb.lb_dns_name
}