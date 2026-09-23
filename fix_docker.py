import os

path = "/DATA/suraj/m1/search_engine/docker-compose.yml"
with open(path, "r") as f:
    content = f.read()

# 1. Line 74
content = content.replace("  api:\n  # ---------------------------------------------------------------------------\n  # 2. Ingestion Worker Service", "  # ---------------------------------------------------------------------------\n  # 2. Ingestion Worker Service")

# 2. postgres service ports
content = content.replace('    ports:\n      - "5432:5432"\n      - "${POSTGRES_PORT:-5432}:5432"', '    ports:\n      - "${POSTGRES_PORT:-5432}:5432"')

# 3. postgres service volumes
content = content.replace('      # Bind mount directly to /DATA/ (5.9 TB free) to avoid exhausting root disk\n      - /DATA/suraj/m1/search_engine/data/postgres_data:/var/lib/postgresql/data\n      - postgres_data:/var/lib/postgresql/data', '      # Bind mount directly to /DATA/ (5.9 TB free) to avoid exhausting root disk\n      - postgres_data:/var/lib/postgresql/data')

# 4. postgres limits
content = content.replace('        limits:\n          cpus: "32.0"\n          memory: 128G\n          cpus: "16.0"\n          memory: 32G', '        limits:\n          cpus: "16.0"\n          memory: 32G')

# 5. postgres healthcheck
content = content.replace('    healthcheck:\n      test: ["CMD-SHELL", "pg_isready -U postgres -d wikirag"]\n      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-wikirag}"]', '    healthcheck:\n      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-wikirag}"]')

# 6. api-service container_name
content = content.replace('    container_name: wikirag_api\n    container_name: wikirag_api_service', '    container_name: wikirag_api_service')

# 7. api-service env vars
content = content.replace('      - DATA_DIR=/DATA/suraj/m1/search_engine/data\n      - HF_HOME=/DATA/suraj/m1/search_engine/data/cache/huggingface\n      - TORCH_HOME=/DATA/suraj/m1/search_engine/data/cache/torch\n      - CUDA_VISIBLE_DEVICES=0\n      - DATA_DIR=/data', '      - DATA_DIR=/data')

# 8. api-service ports
content = content.replace('    ports:\n      - "8000:8000"\n      - "${API_PORT:-8000}:8000"', '    ports:\n      - "${API_PORT:-8000}:8000"')

# 9. api-service volumes
content = content.replace('    volumes:\n      - /DATA/suraj/m1/search_engine/data:/DATA/suraj/m1/search_engine/data\n      - ${DATA_DIR:-/DATA/suraj/m1/search_engine/data}:/data', '    volumes:\n      - ${DATA_DIR:-/DATA/suraj/m1/search_engine/data}:/data')


with open(path, "w") as f:
    f.write(content)

os.system("cp /DATA/suraj/m1/search_engine/docker-compose.yml /DATA/suraj/m1/search_engine/docker/docker-compose.yml")
print("Done")
