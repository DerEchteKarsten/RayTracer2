use anyhow::Result;
use ash::vk::{self, DescriptorSet};
use glam::{IVec2, IVec4, Vec2};
use winit::event::WindowEvent;

use crate::{Buffer, ImageAndView};

enum UIContent {
    Image(usize),
    Text(String),
    TextInput(String),
    NumberInput(f32),
    Checkbox(bool),
}

pub struct UINode {
    pub position: Vec2,
    pub size: Option<IVec2>,
    pub padding: IVec2,
    pub content: UIContent,
    pub children: Vec<UINode>,
    pub event_handler: Box<dyn Fn(UINode, WindowEvent) -> bool>,
}

struct Instantce {
    pub texture_index: u32,
    pub texture_offset: Vec2,
}

pub struct UIPipeline {
    pub pipeline: vk::Pipeline,
    pub descriptor: DescriptorSet,
    pub instance_buffer: Buffer,
    pub font_texture: ImageAndView,
    pub sampler: vk::Sampler,
    pub textures: Vec<ImageAndView>,
}

// fn create_ui_pipeline(ui: UINode) -> Result<UIPipeline> {

//     // Ok(UIPipeline {
//     //     descriptor,
//     //     font_texture,
//     //     instance_buffer,
//     //     pipeline,
//     //     sampler,
//     //     textures,
//     // })
// }
