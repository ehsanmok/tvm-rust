source activate py36

cargo build
cargo test

cd tests/basics
cargo build --features cpu
cargo run --features cpu
if [ $(which nvcc) ]; then
  cargo build --features gpu
  cargo run --features gpu
fi
cd -

cd tests/callback
cargo build
cargo run --bin int
cargo run --bin float
cargo run --bin array
cargo run --bin string
cargo run --bin error
cd -

cd examples/resnet
cargo build
cargo run
cd -

