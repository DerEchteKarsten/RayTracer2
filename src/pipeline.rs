use std::{ffi::CString, io::{self, BufReader}, mem::size_of, path::Path};

use anyhow::Result;
use ash::vk;

use crate::{alinged_size, create_descriptor_set_layout, Buffer, Context};

#[derive(Debug, Clone, Copy, Default)]
pub struct RayTracingShaderGroupInfo {
    pub group_count: u32,
    pub raygen_shader_count: u32,
    pub miss_shader_count: u32,
    pub hit_shader_count: u32,
}

pub struct RayTracingPipeline {
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub shader_binding_table: ShaderBindingTable,
}

pub struct ShaderBindingTable {
    pub raygen_buffer: Buffer,
    pub miss_buffer: Buffer,
    pub hit_buffer: Buffer,

    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

#[derive(Debug, PartialEq)]
pub struct HitGroup {
    pub closest_hit_shader: String,
    pub intersection_shader: Option<String>,
    pub any_hit_shader: Option<String>,
}
#[derive(Debug, PartialEq, Default)]
pub struct RayGenGroup {
    pub raygen_shader: String,
}

#[derive(Debug, PartialEq, Default)]
pub struct MissGroup {
    pub miss_shader: String,
}

#[derive(Debug, PartialEq, Default)]
pub enum ShaderStage {
    Miss,
    #[default] Raygen,
    Intersection,
    ClosestHit,
    AnyHit,
}

#[derive(Debug, PartialEq, Default)]
pub struct DescriptorBinding {
    pub stage: ShaderStage,
    pub _type: vk::DescriptorType,
    pub binding: u32, 
    pub count: u32
}

#[derive(Debug, PartialEq, Default)]
pub struct DescriptorSetBuilder {
    pub bindings: Vec<DescriptorBinding>,
}

impl DescriptorSetBuilder {
    pub fn new() -> Self{
        Default::default()
    }
    pub fn binding(mut self, _type: vk::DescriptorType, stage: ShaderStage) -> DescriptorSetBuilder {
        let binding = self.bindings.len() as u32; 
        self.bindings.push(DescriptorBinding {
            _type,
            stage,
            binding,
            count: 1,
        });
        self
    }
    pub fn array_binding(mut self, _type: vk::DescriptorType, stage: ShaderStage, count: u32) -> DescriptorSetBuilder {
        let binding = self.bindings.len() as u32; 
        self.bindings.push(DescriptorBinding {
            _type,
            stage,
            binding,
            count,
        });
        self
    }
}
#[derive(Debug, Default)]
pub struct RayTracingPipelineBuilder {
    pub hit_groups: Vec<HitGroup>,
    pub miss_groups: Vec<MissGroup>,
    pub raygen_groups: Vec<RayGenGroup>,
    pub sets: Vec<DescriptorSetBuilder>,
    pub push_constants: Vec<vk::PushConstantRange>,
}


pub fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

fn module_from_path(device: &ash::Device, source: String) -> Result<vk::ShaderModule> {
    let file = std::fs::File::open("foo.txt")?;
    let reader = BufReader::new(file);
    let source = reader.buffer();
    let source = read_shader_from_bytes(source)?;

    let create_info = vk::ShaderModuleCreateInfo::builder().code(&source);
    let res = unsafe { device.create_shader_module(&create_info, None) }?;
    Ok(res)
}

fn hit_to_group(group: HitGroup, ctx: &Context, offset: u32) -> (Vec<vk::PipelineShaderStageCreateInfo>, vk::RayTracingShaderGroupCreateInfoKHR, Vec<vk::ShaderModule>) {
    let entry_point_name = CString::new("main").unwrap();
    let mut modules = vec![];
    let mut stages = vec![vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        .module(module_from_path(&ctx.device, group.closest_hit_shader).unwrap())
        .name(&entry_point_name)
        .build()
        ];
    
    let g = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .ty(if group.intersection_shader.is_some() { vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP } else {vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP})
        .general_shader(vk::SHADER_UNUSED_KHR)
        .closest_hit_shader(offset)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR);

    if let Some(intersection) = group.intersection_shader {
        let module = module_from_path(&ctx.device, intersection).unwrap();
        modules.push(module);
        let stage =vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(module)
            .name(&entry_point_name)
            .build();
        g.intersection_shader((stages.len() + 1) as u32);
        stages.push(stage);
    }


    if let Some(any) = group.any_hit_shader {
        let module = module_from_path(&ctx.device, any).unwrap();
        modules.push(module);
        let stage =vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(module)
            .name(&entry_point_name)
            .build();
        g.any_hit_shader((stages.len() + 1) as u32);
        stages.push(stage);
    }
    (
        stages,
        g.build(),
        modules
    )
}

impl RayTracingPipelineBuilder{
    pub fn hit_group(mut self, group: HitGroup) -> RayTracingPipelineBuilder {
        self.hit_groups.push(group);
        self
    }

    pub fn miss_group(mut self, group: MissGroup) -> RayTracingPipelineBuilder {
        self.miss_groups.push(group);
        self
    }

    pub fn raygen_group(mut self, group: RayGenGroup) -> RayTracingPipelineBuilder {
        self.raygen_groups.push(group);
        self
    }

    pub fn descirptor_set(mut self, set: DescriptorSetBuilder) -> RayTracingPipelineBuilder {
        self.sets.push(set);
        self
    }

    pub fn push_constants<A, B, C, D, E>(mut self, ray_gen: Option<A>, miss: Option<B>, chit: Option<C>, anyhit: Option<D>, intersection: Option<E>) -> RayTracingPipelineBuilder {
        if chit.is_some() {
            let range = vk::PushConstantRange::builder()
            .offset(
                if let Some(l) = self.push_constants.last() {
                    l.size + l.offset
                } else {
                    0
                }
            )
            .size(alinged_size(size_of::<B>() as u32, 4))
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build();
            self.push_constants.push(range);
        }
        if miss.is_some() {
            let range = vk::PushConstantRange::builder()
                .offset(
                    if let Some(l) = self.push_constants.last() {
                        l.size + l.offset
                    } else {
                        0
                    }
                )
                .size(alinged_size(size_of::<B>() as u32, 4))
                .stage_flags(vk::ShaderStageFlags::MISS_KHR)
                .build();
            self.push_constants.push(range)
        }
        if ray_gen.is_some() {
            let range = vk::PushConstantRange::builder()
            .offset(
                if let Some(l) = self.push_constants.last() {
                    l.size + l.offset
                } else {
                    0
                }
            )
            .size(alinged_size(size_of::<B>() as u32, 4))
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build();
            self.push_constants.push(range);
        }

        if intersection.is_some() {
            let range = vk::PushConstantRange::builder()
            .offset(
                if let Some(l) = self.push_constants.last() {
                    l.size + l.offset
                } else {
                    0
                }
            )
            .size(alinged_size(size_of::<B>() as u32, 4))
            .stage_flags(vk::ShaderStageFlags::INTERSECTION_KHR)
            .build();
            self.push_constants.push(range);
        }

        if anyhit.is_some() {
            let range = vk::PushConstantRange::builder()
            .offset(
                if let Some(l) = self.push_constants.last() {
                    l.size + l.offset
                } else {
                    0
                }
            )
            .size(alinged_size(size_of::<B>() as u32, 4))
            .stage_flags(vk::ShaderStageFlags::ANY_HIT_KHR)
            .build();
            self.push_constants.push(range);
        }

        self
    }

    fn build(self, ctx: &Context) -> Result<RayTracingPipeline> {
        
        let layouts = self.sets.into_iter().map(|d| {
            let bindings = d.bindings.into_iter().map(|b| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(b.binding)
                    .descriptor_count(b.count)
                    .descriptor_type(b._type)
                    .build()
            }).collect::<Vec<vk::DescriptorSetLayoutBinding>>();
            create_descriptor_set_layout(&ctx.device, bindings.as_slice(), &[]).unwrap()
        }).collect::<Vec<vk::DescriptorSetLayout>>();

        let pipe_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts).push_constant_ranges(&self.push_constants);
        let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipe_layout_info, None)? };

        let mut modules: Vec<vk::ShaderModule> = vec![];
        let mut stages = vec![];
        let mut groups = vec![];

        for h in self.hit_groups {
            let offset = stages.len();
            let (s, g, m) = hit_to_group(h, ctx, offset as u32);
            modules.append(&mut m);
            groups.push(g);
            stages.append(&mut s);
        }
        
       
        let pipe_info = vk::RayTracingPipelineCreateInfoKHR::builder()
            .layout(pipeline_layout)
            .stages(stages.as_slice())
            .groups(groups.as_slice())
            .max_pipeline_ray_recursion_depth(1);

        let inner = unsafe {
            ctx.ray_tracing.pipeline_fn.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipe_info),
                None,
            )?[0]
        };

        Ok(RayTracingPipeline {
            handle: inner,
            descriptor_set_layout: static_dsl,
            layout: pipeline_layout,
            shader_group_info,
            storage_image_set_layout: dynamic_dsl,
        })
    }

}

impl RayTracingPipeline {
    pub fn builder() -> RayTracingPipelineBuilder {
        RayTracingPipelineBuilder::default()
    }
}
