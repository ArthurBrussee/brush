use brush_viewer::viewer::Viewer;
use eframe::NativeOptions;

pub fn create_viewer() -> Viewer {
    let native_options = NativeOptions {
        // Mobile platforms handle viewport differently
        ..Default::default() 
    };
    
    Viewer::new(&eframe::CreationContext {
        options: &native_options,
        ..Default::default()
    })
}
