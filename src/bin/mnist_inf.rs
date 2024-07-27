#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::ClockControl, delay::Delay, peripherals::Peripherals, prelude::*, system::SystemControl,
};

extern crate alloc;
use core::mem::MaybeUninit;

#[global_allocator]
static ALLOCATOR: esp_alloc::EspHeap = esp_alloc::EspHeap::empty();

fn init_heap() {
    const HEAP_SIZE: usize = 32 * 1024;
    static mut HEAP: MaybeUninit<[u8; HEAP_SIZE]> = MaybeUninit::uninit();

    unsafe {
        ALLOCATOR.init(HEAP.as_mut_ptr() as *mut u8, HEAP_SIZE);
    }
}

use burn::backend::ndarray::NdArray;
use burn::tensor;

use espae::mnist::Model;

type Backend = NdArray<f32>;

#[entry]
fn main() -> ! {
    #[cfg(debug_assertions)]
    compile_error!("PSRAM on ESP32 needs to be built in release mode");

    let peripherals = Peripherals::take();
    
    let system = SystemControl::new(peripherals.SYSTEM);

    let clocks = ClockControl::max(system.clock_control).freeze();
    let delay = Delay::new(&clocks);
    init_heap();

    esp_println::logger::init_logger_from_env();

    let device = Default::default();

    let model: Model<Backend> = Model::new(&device);

    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 1, 28, 28], &device);

    let output = model.forward(input);

    log::info!("{:?}", output);

    loop {
        log::info!("Hello world!");
        delay.delay(500.millis());
    }
}
