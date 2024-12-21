use crate::app::{AppContext, AppPanel};
use brush_process::process_loop::ProcessMessage;
use egui::Hyperlink;

pub(crate) struct PresetsPanel {}

impl PresetsPanel {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

fn url_button(label: &str, url: &str, ui: &mut egui::Ui) {
    ui.add(Hyperlink::from_label_and_url(label, url).open_in_new_tab(true));
}

impl AppPanel for PresetsPanel {
    fn title(&self) -> String {
        "Presets".to_owned()
    }

    fn on_message(&mut self, _: &ProcessMessage, _: &mut AppContext) {}

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut AppContext) {
        ui.heading("Mipnerf scenes");

        egui::Grid::new("mip_grid")
            .num_columns(3)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                url_button("bicycle", "https://drive.google.com/file/d/1LawlC-YjHSMl5rwRmEOMQEbJUioaYI5p/view?usp=drive_link", ui);
                url_button("bonsai", "https://drive.google.com/file/d/1IWhmM49q_pfUZzJhA_vXv4POBODSAh32/view?usp=drive_link", ui);
                url_button("counter", "https://drive.google.com/file/d/1564FHRsObZDGUlRx4RTFBTCi8jDPzTjj/view?usp=drive_link", ui);
                ui.end_row();

                url_button("garden", "https://drive.google.com/file/d/1WROBCrVu3YqA60mbRGmSRYXOJB4N5KAk/view?usp=drive_link", ui);
                url_button("kitchen", "https://drive.google.com/file/d/1VSJM4b3pcQYiZj4xWSIIzHhwbzMcFWZv/view?usp=drive_link", ui);
                url_button("room", "https://drive.google.com/file/d/1ieRBqlouADIAbCy8ryjI7M2PsfSNR23u/view?usp=drive_link", ui);
                ui.end_row();

                url_button("stump", "https://drive.google.com/file/d/1noPG4AowuT__xFV4uHODzOW7te9Kbb-T/view?usp=drive_link", ui);
                ui.end_row();
            });

        ui.heading("Synthetic blender scenes");
        egui::Grid::new("blend_grid")
            .num_columns(4)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                url_button("chair", "https://drive.google.com/file/d/1EUcmoo5c2Ab9SiyWc8dZxbOxkEKWTU4C/view?usp=drive_link", ui);
                url_button("drums", "https://drive.google.com/file/d/1UpBQoUJ9ShKgsyM7WaPy0a6qqtUMSOCx/view?usp=drive_link", ui);
                url_button("ficus", "https://drive.google.com/file/d/1hwE1z0GSRHfMGXx3TyhuyqT-pDReeRik/view?usp=drive_link", ui);
                ui.end_row();

                url_button("hotdog", "https://drive.google.com/file/d/1EtIyCOyFAbTKHlMvNSwCFr5C1peyI107/view?usp=drive_link", ui);
                url_button("lego", "https://drive.google.com/file/d/16TY5KxWUq7OzjkkLDBGNKZ0P5Laf-oaL/view?usp=drive_link", ui);
                url_button("materials", "https://drive.google.com/file/d/1MWxV_NReK-UW4zKMbDIxQNiPwALZGSpd/view?usp=drive_link", ui);
                ui.end_row();

                url_button("mic", "https://drive.google.com/file/d/1s1PpJe71OECKnrUeNVdzjhKk-JXKlngI/view?usp=drive_link", ui);
                url_button("ship", "https://drive.google.com/file/d/1Wvne6m7voRj8LvSosvq9vKMp8UYMCrER/view?usp=drive_link", ui);
            });
    }
}
