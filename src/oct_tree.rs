use anyhow::Result;
use glam::UVec3;

#[derive(Debug, Default)]
pub struct Octant {
    pub children: Option<Box<[Octant; 8]>>,
    pub color: Option<u32>,
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
            if self.color.is_some() {
                println!("Override")
            } else {
                self.color = Some(color);
            }
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
    
    pub fn load(path: &str) -> Result<Self> {
        let mut octree = Self::default();
        let file = dot_vox::load(path).unwrap();
        let models = file.models;
        for (_, m) in models.iter().enumerate() {
            let mut size = m.size.x;
            if size < m.size.y {
                size = m.size.y
            }
            if size < m.size.z {
                size = m.size.z
            }
            size = 2_u32.pow(((size as f32).log2()).ceil() as u32);
            for v in m.voxels.iter() {
                let color = file.palette[v.i as usize];
                let u32_color = ((color.b as u32) << 16) | ((color.g as u32) << 8) | (color.r as u32);
                let pos = UVec3::new(size - v.x as u32, v.z as u32, v.y as u32);
                octree.append_voxel(u32_color, size, pos);
            }
        }
    
        return Ok(octree);
    }
}
