use burn_import::onnx::{ModelGen, RecordType};

fn main() {
    println!("cargo:rustc-link-arg-bins=-Tlinkall.x");

    println!("cargo:rustc-link-arg-bins=-Trom_functions.x");
    
    ModelGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();
}
