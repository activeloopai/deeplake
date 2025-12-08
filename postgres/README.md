# pg_deeplake

PostgreSQL extension for vector similarity search, full-text search, and hybrid search using DeepLake.

## Quick Start with Docker

### Installation

```bash
# Pull and run the Docker container
docker run -d \
  --name pg-deeplake \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  quay.io/activeloopai/pg-deeplake:18
```

### Connect

```bash
# Connect using psql
psql -h localhost -p 5432 -U postgres
```

### Basic Usage

```sql
-- 1. Enable extension
CREATE EXTENSION pg_deeplake;

-- 2. Create a table with DeepLake storage
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    v1 float4[],
    v2 float4[]
) USING deeplake;

-- 3. Create an index
CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC);

-- 4. Insert data
INSERT INTO vectors (v1, v2) VALUES
    (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]),
    (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]),
    (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]);

-- 5. Query with cosine similarity
SELECT id, v1 <#> ARRAY[1.0, 2.0, 3.0] AS score
FROM vectors
ORDER BY score DESC
LIMIT 10;
```

### What You Get

- **Vector similarity search** using `<#>` operator (cosine similarity)
- **Full-text search** with BM25 scoring on text columns
- **Hybrid search** combining vectors and text
- **Numeric indexing** for integer/float columns
- **JSON indexing** for JSONB fields

---

## Building from Source

### Requirements

- pkg-config
- autoconf-archive
- libglew
- libiconv
- CMake 3.29.2 or upper
- Clang 16 or upper
- vcpkg

For building from python:
- python-venv
- python-dev (e.g., python3.10-dev)

### Installing Dependencies

```bash
sudo apt update
sudo apt install curl zip unzip tar -y
sudo apt install pkg-config -y
sudo apt install autoconf-archive -y
sudo apt install make -y
sudo apt install flex -y
sudo apt install bison -y
sudo apt install libglew-dev -y

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18  # For Clang 18
sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang-18 100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-18 100

wget https://ftp.gnu.org/gnu/libiconv/libiconv-1.17.tar.gz
tar -xzf libiconv-1.17.tar.gz
cd libiconv-1.17
./configure --prefix=/usr/local
make
sudo make install
cd ../

wget https://github.com/Kitware/CMake/releases/download/v4.0.5/cmake-4.0.5-linux-x86_64.sh
chmod +x cmake-4.0.5-linux-x86_64.sh
./cmake-4.0.5-linux-x86_64.sh
cd cmake-4.0.5-linux-x86_64/bin/
echo "export PATH=$PATH:`pwd`" >> ~/.bashrc # or path to your shell config file
cd ../../
```

### vcpkg Configuration

VCPkg is installed via a git repository which a VCPKG_ROOT environment variable points to.

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg/
export VCPKG_ROOT=`pwd`
git checkout 6f29f12e82a8293156836ad81cc9bf5af41fe836
./bootstrap-vcpkg.sh
echo "export VCPKG_ROOT=$VCPKG_ROOT" >> ~/.bashrc # or path to your shell config file
echo "export PATH=$PATH:$VCPKG_ROOT" >> ~/.bashrc # or path to your shell config file
source ~/.bashrc
cd ../
```

### Build the Extension

```bash
python3 scripts/build_pg_ext.py [debug|dev|prod]
```

### Run Tests

```bash
cd postgres/tests
make test
```
