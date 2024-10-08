use ash::vk;
use bevy::prelude::*;
use gpu_allocator::MemoryLocation;

use crate::WINDOW_SIZE;

use super::renderer::{Buffer, ImageAndView, Renderer};

#[derive(Resource)]
pub(super) struct RenderResources {
    pub(super) g_buffer: [ImageAndView; 2],
    pub(super) hash_map_buffer: Buffer,
}

impl RenderResources {
    pub(super) fn new(ctx: &mut Renderer) -> Self {
        let mut g_buffer = [ImageAndView::default(), ImageAndView::default()];

        for buffer in g_buffer.iter_mut() {
            let image = ctx
                .create_image(
                    vk::ImageUsageFlags::STORAGE,
                    MemoryLocation::GpuOnly,
                    vk::Format::R32_UINT,
                    WINDOW_SIZE.0,
                    WINDOW_SIZE.1,
                )
                .unwrap();
            let view = ctx.create_image_view(&image).unwrap();
            buffer.image = image;
            buffer.view = view;
        }

        let hash_map_buffer = ctx
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                2073600 * 22,
                Some("hashmap_buffer"),
            )
            .unwrap();

        Self {
            g_buffer,
            hash_map_buffer,
        }
    }
}
