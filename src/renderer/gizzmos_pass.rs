
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Gizzmo {
    pub color: glam::Vec4,
    pub position: glam::Vec3,
    pub radius: f32,
}

#[derive(Resource)]
pub struct GizzmoBuffer {
    pub staging: Buffer,
    pub buffer: Buffer,
    pub data: [Gizzmo; 10],
    pub dirty: bool,
}

impl GizzmoBuffer {
    pub fn sphere(&mut self, i: usize, color: glam::Vec3, pos: glam::Vec3, radius: f32) {
        self.data[i as usize].color = glam::vec4(color.x, color.y, color.z, 1.0);
        self.data[i as usize].position = pos;
        self.data[i as usize].radius = radius;
        self.dirty = true;
    }

    pub fn update(&mut self, renderer: &mut Renderer) {
        self.staging.copy_data_to_buffer(&self.data).unwrap();
        renderer
            .execute_one_time_commands(|cmd_buffer| {
                renderer.copy_buffer(cmd_buffer, &self.staging, &self.buffer);
            })
            .unwrap();
        // info!("{:?}", self.data);
        self.dirty = false;
    }
}