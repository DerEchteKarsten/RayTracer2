use anyhow::Result;
use bevy::prelude::Resource;
use dot_vox::{DotVoxData, SceneNode};
use glam::{ivec3, uvec3, vec3, IVec3, Mat3, UVec3, Vec3};

#[derive(Debug, Default)]
pub struct Octant {
    pub children: Option<Box<[Octant; 8]>>,
    pub color: Option<u32>,
}

#[derive(Resource)]
pub struct GameWorld {
    pub tree: Octant,
    pub tree_level: u32,
    pub level_dim: u32,
    pub build_tree: Vec<u32>,
}

impl Clone for Octant {
    fn clone(&self) -> Self {
        Octant {
            children: match self.children.as_ref() {
                None => None,
                Some(children) => Some(children.clone()),
            },
            color: self.color,
        }
    }
}

pub fn ray_box(origin: glam::Vec3, dir: glam::Vec3, lb: glam::Vec3, rt: glam::Vec3) -> f32 {
    // r.dir is unit direction vector of ray
    let dirfrac = vec3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    let t1 = (lb.x - origin.x) * dirfrac.x;
    let t2 = (rt.x - origin.x) * dirfrac.x;
    let t3 = (lb.y - origin.y) * dirfrac.y;
    let t4 = (rt.y - origin.y) * dirfrac.y;
    let t5 = (lb.z - origin.z) * dirfrac.z;
    let t6 = (rt.z - origin.z) * dirfrac.z;

    let tmin = f32::max(
        f32::max(f32::min(t1, t2), f32::min(t3, t4)),
        f32::min(t5, t6),
    );
    let tmax = f32::min(
        f32::min(f32::max(t1, t2), f32::max(t3, t4)),
        f32::max(t5, t6),
    );

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if tmax < 0.0 {
        return f32::INFINITY;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax {
        return f32::INFINITY;
    }

    return tmin;
}

impl Octant {
    pub fn _count(&self, depth: u32, total: &mut u32) {
        if let Some(color) = self.color {
            *total += 1;
            println!("Color: {}", color);
        }
        if depth == 0 {
            return;
        }
        if let Some(children) = self.children.as_ref() {
            for i in 0..8 {
                children[i as usize]._count(depth - 1, total);
            }
        }
        return;
    }

    pub fn _filter<T>(&self, candidates: &mut Vec<Octant>, predicate: &T)
    where
        T: Fn(&Octant) -> bool,
    {
        if predicate(self) {
            candidates.push(self.clone());
        }
        if let Some(children) = self.children.as_ref() {
            for i in 0..8 {
                children[i as usize]._filter(candidates, predicate);
            }
        }
        return;
    }

    pub fn append_voxel(&mut self, color: u32, level_dim: u32, level_pos: UVec3) {
        if level_dim <= 1 {
            self.color = Some(color);
            return;
        }

        let level_dim = level_dim >> 1;
        let cmp = level_pos.cmpge(UVec3::new(level_dim, level_dim, level_dim));
        let child_slot_index = cmp.x as u32 | (cmp.y as u32) << 1 | (cmp.z as u32) << 2;
        let new_pos = level_pos
            - (UVec3::new(
                cmp.x as u32 * level_dim,
                cmp.y as u32 * level_dim,
                cmp.z as u32 * level_dim,
            ));

        match self.children.as_mut() {
            None => {
                let mut children: [Octant; 8] = Default::default();

                children[child_slot_index as usize].append_voxel(color, level_dim, new_pos);
                self.children = Some(Box::new(children));
            }
            Some(children) => {
                children[child_slot_index as usize].append_voxel(color, level_dim, new_pos);
            }
        }
    }

    pub fn find_voxel(&self, level_pos: UVec3, level_dim: u32) -> Option<u32> {
        if level_dim <= 1 {
            return self.color;
        }

        let level_dim = level_dim >> 1;
        let cmp = level_pos.cmpge(UVec3::new(level_dim, level_dim, level_dim));
        let child_slot_index = cmp.x as u32 | (cmp.y as u32) << 1 | (cmp.z as u32) << 2;
        let new_pos = level_pos
            - (UVec3::new(
                cmp.x as u32 * level_dim,
                cmp.y as u32 * level_dim,
                cmp.z as u32 * level_dim,
            ));

        if let Some(children) = self.children.as_ref() {
            children[child_slot_index as usize].find_voxel(new_pos, level_dim)
        } else {
            None
        }
    }

    pub fn build(&self) -> Vec<u32> {
        let mut tree = vec![];

        let mut candidates = vec![self];
        let mut new_candidates = vec![];
        for _ in 0..10 {
            let i = tree.len() as u32 + candidates.len() as u32 * 8;
            for c in &candidates {
                for child in c.children.as_ref().unwrap().iter() {
                    let node = if child.children.is_some() {
                        let candidate = new_candidates.len() as u32;
                        new_candidates.push(child);
                        0x8000_0000 + i + candidate * 8
                    } else {
                        if let Some(color) = child.color {
                            0xC000_0000 | color
                        } else {
                            0x0000_0000
                        }
                    };
                    tree.push(node);
                }
            }

            candidates = new_candidates;
            new_candidates = vec![]
        }
        tree
    }

    pub fn load(path: &str) -> Result<(Self, u32)> {
        let mut octree = Self::default();
        let mut file = dot_vox::load(path).unwrap();
        let size = 1024;
        let node = file.scenes.first().unwrap().clone();
        traverse_scene(&node, &mut file, &mut octree, size, Vec3::ZERO, None);
        // let mut size = m.size.x;
        // if size < m.size.y {
        //     size = m.size.y
        // }
        // if size < m.size.z {
        //     size = m.size.z
        // }
        // size = 2_u32.pow(((size as f32).log2()).ceil() as u32);
        
        return Ok((octree, size));
    }

    pub fn depth(&self, current_depth: u32) -> u32 {
        if let Some(children) = &self.children {
            let mut maxdepth = 0;
            for i in children.as_ref() {
                let depth = i.depth(current_depth + 1);
                if depth > maxdepth {
                    maxdepth = depth;
                }
            }
            return maxdepth;
        }else {
            return current_depth;
        }
    }

    pub fn trace(
        &self,
        org: glam::Vec3,
        dir: glam::Vec3,
        level_dim: u32,
        level_pos: Vec3,
        depth: &mut f32,
    ) {
        let box_hit = ray_box(
            org,
            dir,
            level_pos - Vec3::splat((level_dim) as f32),
            level_pos + Vec3::splat((level_dim) as f32),
        );
        if box_hit == f32::INFINITY || box_hit > *depth {
            return;
        }
        if self.color.is_some() {
            *depth = box_hit;
        }
        if level_dim <= 1 || (self.color.is_none() && self.children.is_none()) {
            return;
        }
        let level_dim = level_dim / 4;
        if let Some(children) = self.children.as_ref() {
            for (i, c) in children.iter().enumerate() {
                c.trace(
                    org,
                    dir,
                    level_dim,
                    level_pos
                        + (level_dim as f32)
                            * vec3(
                                (i & 0b001 == 1) as u32 as f32,
                                (i & 0b010 == 1) as u32 as f32,
                                (i & 0b100 == 1) as u32 as f32,
                            ),
                    depth,
                );
            }
        }
    }
}

fn traverse_scene(
    node: &SceneNode,
    file: &mut DotVoxData,
    octree: &mut Octant,
    size: u32,
    transform: Vec3,
    rotation: Option<u8>,
) {
    match node {
        SceneNode::Group {
            attributes,
            children,
        } => {
            for c in children {
                let child_node = file.scenes[*c as usize].clone();
                traverse_scene(&child_node, file, octree, size, transform, rotation)
            }
        }
        SceneNode::Shape { attributes, models } => {
            for m in models {
                let model = &file.models[m.model_id as usize];
                for v in model.voxels.iter() {
                    let color = file.palette[v.i as usize];
                    let u32_color =
                        ((color.b as u32) << 16) | ((color.g as u32) << 8) | (color.r as u32);
                    let mut pos = Vec3::new(v.x as f32, v.z as f32, v.y as f32)
                        - (vec3(
                            model.size.x as f32,
                            model.size.z as f32,
                            model.size.y as f32,
                        ) / 2.0);
                    if let Some(rotation) = rotation {
                        let mut arry = [0_f32; 9];
                        let mut mask = 0u8;
                        arry[(rotation & 0b00000_0011) as usize] = 1.0;
                        mask |= match rotation & 0b0000_0011 {
                            0 => 0b1,
                            1 => 0b10,
                            2 => 0b100,
                            _ => 0,
                        };
                        arry[(rotation & 0b00000_1100) as usize + 3] = 1.0;
                        mask |= match rotation & 0b0000_1100 {
                            0 => 0b1,
                            1 => 0b10,
                            2 => 0b100,
                            _ => 0,
                        };
                        arry[mask.trailing_ones() as usize + 6] = 1.0;

                        let rot_mat = Mat3::from_cols_array(&arry);
                        let rot_mat = Mat3::from_cols(
                            rot_mat.col(2)
                                * if (rotation & 0b0100_0000u8) == 1 {
                                    -1.0
                                } else {
                                    1.0
                                },
                            rot_mat.col(0)
                                * if (rotation & 0b0001_0000u8) == 1 {
                                    -1.0
                                } else {
                                    1.0
                                },
                            rot_mat.col(1)
                                * if (rotation & 0b0010_0000u8) == 1 {
                                    1.0
                                } else {
                                    -1.0
                                },
                        )
                        .transpose();
                        pos = rot_mat * pos;
                    }
                    pos += transform;
                    octree.append_voxel(
                        u32_color,
                        size,
                        (pos + vec3(size as f32 / 2.0, size as f32 / 2.0, size as f32 / 2.0))
                            .as_uvec3(),
                    );
                }
            }
        }
        SceneNode::Transform {
            attributes,
            frames,
            child,
            layer_id,
        } => {
            let child_node = &file.scenes[(*child) as usize].clone();
            let attribs = &frames.first().unwrap().attributes;
            let attribs_val = attribs.get("_t");
            let transform_string = attribs_val.cloned().unwrap_or("0 0 0".to_owned());
            let components = transform_string
                .split(" ")
                .map(|s| s.parse::<i32>().unwrap_or(0))
                .collect::<Vec<_>>();
            let rot = attribs
                .get("_r")
                .cloned()
                .and_then(|r| r.parse::<u8>().ok());
            let parsed_transform = Vec3::new(
                -components[0] as f32,
                components[2] as f32,
                components[1] as f32,
            );
            // println!("{:?}, {}", attribs, rot);
            traverse_scene(
                child_node,
                file,
                octree,
                size,
                transform + parsed_transform,
                rot,
            );
        }
    }
}

const STACK_SIZE: u32 = 23;
const EPS: f32 = 3.552713678800501e-15;

#[derive(Clone, Copy)]
struct StackItem {
    node: u32,
    t_max: f32,
}

pub fn ray_voxel(uOctree: &Vec<u32>, o: Vec3, d: Vec3) -> Option<(Vec3, Vec3)> {
    let mut d = d;
    let mut stack = [StackItem {
        node: 0,
        t_max: 0.0,
    }; (STACK_SIZE + 1) as usize];
    d.x = if f32::abs(d.x) >= EPS {
        d.x
    } else {
        if d.x >= 0.0 {
            EPS
        } else {
            -EPS
        }
    };
    d.y = if f32::abs(d.y) >= EPS {
        d.y
    } else {
        if d.y >= 0.0 {
            EPS
        } else {
            -EPS
        }
    };
    d.z = if f32::abs(d.z) >= EPS {
        d.z
    } else {
        if d.z >= 0.0 {
            EPS
        } else {
            -EPS
        }
    };

    // Precompute the coefficients of tx(x), ty(y), and tz(z).
    // The octree is assumed to reside at coordinates [1, 2].
    let t_coef = 1.0 / -Vec3::abs(d);
    let mut t_bias = t_coef * o;

    let mut oct_mask = 0_u32;
    if d.x > 0.0 {
        oct_mask ^= 1_u32;
        t_bias.x = 3.0 * t_coef.x - t_bias.x;
    }
    if d.y > 0.0 {
        oct_mask ^= 2_u32;
        t_bias.y = 3.0 * t_coef.y - t_bias.y;
    }
    if d.z > 0.0 {
        oct_mask ^= 4_u32;
        t_bias.z = 3.0 * t_coef.z - t_bias.z;
    }

    // Initialize the active span of t-values.
    let mut t_min = f32::max(
        f32::max(2.0 * t_coef.x - t_bias.x, 2.0 * t_coef.y - t_bias.y),
        2.0 * t_coef.z - t_bias.z,
    );
    let mut t_max = f32::min(
        f32::min(t_coef.x - t_bias.x, t_coef.y - t_bias.y),
        t_coef.z - t_bias.z,
    );
    t_min = f32::max(t_min, 0.0);
    let mut h = t_max;

    let mut parent = 0_u32;
    let mut cur = 0_u32;
    let mut pos = Vec3::ONE;
    let mut child_slot_index = 0_u32;
    if 1.5 * t_coef.x - t_bias.x > t_min {
        child_slot_index ^= 1_u32;
        pos.x = 1.5;
    }
    if 1.5 * t_coef.y - t_bias.y > t_min {
        child_slot_index ^= 2_u32;
        pos.y = 1.5;
    }
    if 1.5 * t_coef.z - t_bias.z > t_min {
        child_slot_index ^= 4_u32;
        pos.z = 1.5;
    }

    let mut scale = STACK_SIZE - 1;
    let mut scale_exp2 = 0.5; // exp2( scale - STACK_SIZE )

    while scale < STACK_SIZE {
        if cur == 0_u32 {
            cur = uOctree[(parent + (child_slot_index ^ oct_mask)) as usize];
        }
        // Determine maximum t-value of the cube by evaluating
        // tx(), ty(), and tz() at its corner.

        let t_corner = pos * t_coef - t_bias;
        let tc_max = f32::min(f32::min(t_corner.x, t_corner.y), t_corner.z);

        if (cur & 0x80000000_u32) != 0_u32 && t_min <= t_max {
            // INTERSECT
            let tv_max = f32::min(t_max, tc_max);
            let half_scale_exp2 = scale_exp2 * 0.5;
            let t_center = half_scale_exp2 * t_coef + t_corner;

            if t_min <= tv_max {
                if (cur & 0x40000000) != 0_u32 {
                    break;
                }
                // leaf node

                // PUSH
                if tc_max < h {
                    stack[scale as usize].node = parent;
                    stack[scale as usize].t_max = t_max;
                }
                h = tc_max;

                parent = cur & 0x3fffffff_u32;

                child_slot_index = 0_u32;
                scale -= 1_u32;
                scale_exp2 = half_scale_exp2;
                if t_center.x > t_min {
                    child_slot_index ^= 1_u32;
                    pos.x += scale_exp2;
                }
                if t_center.y > t_min {
                    child_slot_index ^= 2_u32;
                    pos.y += scale_exp2;
                }
                if t_center.z > t_min {
                    child_slot_index ^= 4_u32;
                    pos.z += scale_exp2;
                }

                cur = 0_u32;
                t_max = tv_max;

                continue;
            }
        }

        // ADVANCE
        let mut step_mask = 0_u32;
        if t_corner.x <= tc_max {
            step_mask ^= 1_u32;
            pos.x -= scale_exp2;
        }
        if t_corner.y <= tc_max {
            step_mask ^= 2_u32;
            pos.y -= scale_exp2;
        }
        if t_corner.z <= tc_max {
            step_mask ^= 4_u32;
            pos.z -= scale_exp2;
        }

        // Update active t-span and flip bits of the child slot index.
        t_min = tc_max;
        child_slot_index ^= step_mask;

        // Proceed with pop if the bit flips disagree with the ray direction.
        if (child_slot_index & step_mask) != 0_u32 {
            unsafe {
                // POP
                // Find the highest differing bit between the two positions.
                let mut differing_bits: u32 = 0_u32;
                if (step_mask & 1_u32) != 0_u32 {
                    differing_bits |= std::mem::transmute::<f32, u32>(pos.x)
                        ^ std::mem::transmute::<f32, u32>(pos.x + scale_exp2);
                }
                if (step_mask & 2_u32) != 0_u32 {
                    differing_bits |= std::mem::transmute::<f32, u32>(pos.y)
                        ^ std::mem::transmute::<f32, u32>(pos.y + scale_exp2);
                }
                if (step_mask & 4_u32) != 0_u32 {
                    differing_bits |= std::mem::transmute::<f32, u32>(pos.z)
                        ^ std::mem::transmute::<f32, u32>(pos.z + scale_exp2);
                }
                scale = 31_u32 - differing_bits.leading_zeros();
                scale_exp2 = std::mem::transmute::<u32, f32>(
                    (scale.wrapping_sub(STACK_SIZE).wrapping_add(127u32)) << 23u32,
                ); // exp2f(scale - s_max)

                // Restore parent voxel from the stack.
                parent = stack[scale as usize].node;
                t_max = stack[scale as usize].t_max;

                // Round cube position and extract child slot index.
                let shx: u32 = std::mem::transmute::<f32, u32>(pos.x) >> scale;
                let shy: u32 = std::mem::transmute::<f32, u32>(pos.y) >> scale;
                let shz: u32 = std::mem::transmute::<f32, u32>(pos.z) >> scale;
                pos.x = std::mem::transmute::<u32, f32>(shx << scale);
                pos.y = std::mem::transmute::<u32, f32>(shy << scale);
                pos.z = std::mem::transmute::<u32, f32>(shz << scale);
                child_slot_index =
                    (shx & 1_u32) | ((shy & 1_u32) << 1_u32) | ((shz & 1_u32) << 2_u32);

                // Prevent same parent from being stored again and invalidate cached
                // child descriptor.
                h = 0.0;
                cur = 0_u32;
            }
        }
    }

    let mut norm;
    let t_corner = t_coef * (pos + scale_exp2) - t_bias;
    if t_corner.x > t_corner.y && t_corner.x > t_corner.z {
        norm = vec3(-1.0, 0.0, 0.0);
    } else if t_corner.y > t_corner.z {
        norm = vec3(0.0, -1.0, 0.0);
    } else {
        norm = vec3(0.0, 0.0, -1.0);
    }

    if (oct_mask & 1) == 0_u32 {
        norm.x = -norm.x;
    }
    if (oct_mask & 2) == 0_u32 {
        norm.y = -norm.y;
    }
    if (oct_mask & 4) == 0_u32 {
        norm.z = -norm.z;
    }

    // Undo mirroring of the coordinate system.
    if (oct_mask & 1_u32) != 0 {
        pos.x = 3.0 - scale_exp2 - pos.x;
    }
    if (oct_mask & 2_u32) != 0 {
        pos.y = 3.0 - scale_exp2 - pos.y;
    }
    if (oct_mask & 4_u32) != 0 {
        pos.z = 3.0 - scale_exp2 - pos.z;
    }

    // Output results.
    let mut o_pos = Vec3::clamp(o + t_min * d, pos, pos + scale_exp2);
    if norm.x != 0.0 {
        o_pos.x = if norm.x > 0.0 {
            pos.x + scale_exp2 + EPS * 2.0
        } else {
            pos.x - EPS
        };
    }
    if norm.y != 0.0 {
        o_pos.y = if norm.y > 0.0 {
            pos.y + scale_exp2 + EPS * 2.0
        } else {
            pos.y - EPS
        };
    }
    if norm.z != 0.0 {
        o_pos.z = if norm.z > 0.0 {
            pos.z + scale_exp2 + EPS * 2.0
        } else {
            pos.z - EPS
        };
    }

    if scale < STACK_SIZE && t_min <= t_max && o_pos.y != 1.0 {
        Some((o_pos, norm))
    } else {
        None
    }
}
