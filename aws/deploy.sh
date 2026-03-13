#!/bin/bash
# ── SupplyGuard AI — AWS Deployment Script ────────────────────────
# Builds Docker image, pushes to ECR, deploys to ECS Fargate.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed and running
#   - .env file populated
#   - IAM user with: ECR:*, ECS:*, EC2:DescribeVpcs, IAM:PassRole
#
# Usage: chmod +x aws/deploy.sh && ./aws/deploy.sh

set -euo pipefail

# ── Load env ─────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

: "${AWS_REGION:?AWS_REGION not set}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID not set}"
: "${ECR_REPO:=supplyguard-ai}"
: "${ECS_CLUSTER:=supplyguard-cluster}"
: "${ECS_SERVICE:=supplyguard-service}"

IMAGE_TAG="${ECR_REPO}:$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "═══════════════════════════════════════════"
echo "  SupplyGuard AI — AWS Deployment"
echo "  Region : ${AWS_REGION}"
echo "  Image  : ${ECR_URI}:latest"
echo "═══════════════════════════════════════════"

# ── 1. Create ECR repo (idempotent) ──────────────────────────────
echo "→ Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 \
    || aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        > /dev/null

# ── 2. Docker login to ECR ────────────────────────────────────────
echo "→ Authenticating with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ECR_URI}"

# ── 3. Build image ────────────────────────────────────────────────
echo "→ Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t "${ECR_REPO}:latest" \
    -t "${IMAGE_TAG}" \
    .

# ── 4. Tag & push ─────────────────────────────────────────────────
echo "→ Pushing to ECR..."
docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"
docker tag "${IMAGE_TAG}" "${ECR_URI}:$(echo ${IMAGE_TAG} | cut -d: -f2)"
docker push "${ECR_URI}:latest"
docker push "${ECR_URI}:$(echo ${IMAGE_TAG} | cut -d: -f2)"

# ── 5. Register ECS Task Definition ──────────────────────────────
echo "→ Registering ECS Task Definition..."

TASK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/supplyguard-ecs-task-role"
EXEC_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole"

TASK_DEF=$(cat <<EOF
{
  "family": "supplyguard-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "${EXEC_ROLE_ARN}",
  "taskRoleArn": "${TASK_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "supplyguard",
      "image": "${ECR_URI}:latest",
      "portMappings": [
        {"containerPort": 8501, "protocol": "tcp"},
        {"containerPort": 8001, "protocol": "tcp"}
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/supplyguard",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "supplyguard"
        }
      },
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:supplyguard/anthropic_api_key"
        },
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:supplyguard/secret_key"
        },
        {
          "name": "API_KEY_ADMIN",
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:supplyguard/api_key_admin"
        }
      ],
      "environment": [
        {"name": "AWS_REGION", "value": "${AWS_REGION}"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8001/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF
)

TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json "${TASK_DEF}" \
    --region "${AWS_REGION}" \
    --query "taskDefinition.taskDefinitionArn" \
    --output text)

echo "  Task definition: ${TASK_DEF_ARN}"

# ── 6. Create CloudWatch log group ────────────────────────────────
aws logs create-log-group \
    --log-group-name "/ecs/supplyguard" \
    --region "${AWS_REGION}" 2>/dev/null || true

# ── 7. Update ECS Service ─────────────────────────────────────────
echo "→ Updating ECS Service..."
aws ecs update-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${ECS_SERVICE}" \
    --task-definition "${TASK_DEF_ARN}" \
    --force-new-deployment \
    --region "${AWS_REGION}" \
    > /dev/null

echo "→ Waiting for service to stabilise (this takes 2-5 minutes)..."
aws ecs wait services-stable \
    --cluster "${ECS_CLUSTER}" \
    --services "${ECS_SERVICE}" \
    --region "${AWS_REGION}"

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo "  Streamlit UI : https://your-alb-domain.com"
echo "  FastAPI Docs : https://your-alb-domain.com/api/docs"
echo "═══════════════════════════════════════════"
