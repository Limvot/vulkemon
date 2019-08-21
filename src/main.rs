
use std::fs::File;
use std::collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use failure::Error;

use serde::{Serialize, Deserialize};
use serde_json;

use wgpu::winit::*;
use image::*;

struct Square {
    pub x: f32,
    pub y: f32,
    pub sx: f32,
    pub sy: f32,
    pub tx: f32,
    pub ty: f32,
    pub stx: f32,
    pub sty: f32,
    uniform_size: u64,
    uniform_buf: wgpu::Buffer,
    uniform_buf_dirty: bool,
    pub local_bind_group: wgpu::BindGroup,
}
impl Square {
    fn new(x: f32, y: f32, sx: f32, sy: f32, tx: f32, ty: f32, stx: f32, sty: f32, device: &wgpu::Device, local_bind_group_layout: &wgpu::BindGroupLayout) -> Square {
        let uniform_buf = device.create_buffer_mapped(8, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST).fill_from_slice(&[x, y, sx, sy, tx, ty, stx, sty]);
        let uniform_size = 4*8;
        let local_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: local_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0 .. uniform_size,
                    },
                },
            ],
        });
        Square {
            x, y, sx, sy, tx, ty, stx, sty, uniform_size, uniform_buf, uniform_buf_dirty: false, local_bind_group,
        }
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
        self.uniform_buf_dirty = true;
    }
    fn pos_delta(&mut self, dx: f32, dy: f32) {
        self.x += dx;
        self.y += dy;
        self.uniform_buf_dirty = true;
    }
    fn set_siz(&mut self, sx: f32, sy: f32) {
        self.sx = sx;
        self.sy = sy;
        self.uniform_buf_dirty = true;
    }
    fn siz_delta(&mut self, dx: f32, dy: f32) {
        self.sx += dx;
        self.sy += dy;
        self.uniform_buf_dirty = true;
    }
    fn maybe_update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.uniform_buf_dirty {
            let updated_uniform_buf = device.create_buffer_mapped(8, wgpu::BufferUsage::TRANSFER_SRC).fill_from_slice(&[self.x, self.y, self.sx, self.sy, self.tx, self.ty, self.stx, self.sty]);
            encoder.copy_buffer_to_buffer(&updated_uniform_buf, 0, &self.uniform_buf, 0, self.uniform_size);
        }
    }
}

struct World {
    x: f32,
    y: f32,
    scale: f32,
    uniform_size: u64,
    uniform_buf: wgpu::Buffer,
    uniform_buf_dirty: bool,
    pub world_bind_group: wgpu::BindGroup,
}
impl World {
    fn new(x: f32, y: f32, scale: f32, device: &wgpu::Device, world_bind_group_layout: &wgpu::BindGroupLayout) -> World {
        let uniform_buf = device.create_buffer_mapped(4, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST).fill_from_slice(&[x, y, scale, scale]);
        let uniform_size = 4*4;
        let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: world_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0 .. uniform_size,
                    },
                },
            ],
        });
        World {
            x, y, scale, uniform_size, uniform_buf, uniform_buf_dirty: false, world_bind_group,
        }
    }
    fn pos_delta(&mut self, dx: f32, dy: f32) {
        self.x += dx;
        self.y += dy;
        self.uniform_buf_dirty = true;
    }
    fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }
    fn scale_up(&mut self) {
        self.scale *= 1.0/0.9;
        self.uniform_buf_dirty = true;
    }
    fn scale_down(&mut self) {
        self.scale *= 0.9;
        self.uniform_buf_dirty = true;
    }
    fn maybe_update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.uniform_buf_dirty {
            let updated_uniform_buf = device.create_buffer_mapped(4, wgpu::BufferUsage::TRANSFER_SRC).fill_from_slice(&[self.x, self.y, self.scale, self.scale]);
            encoder.copy_buffer_to_buffer(&updated_uniform_buf, 0, &self.uniform_buf, 0, self.uniform_size);
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Save {
    x_increment: f32,
    y_increment: f32,
    x_start: f32,
    y_start: f32,
    width_in_tiles:  usize,
    height_in_tiles: usize,
    hash_to_type: BTreeMap<u64, f32>,
}

fn main() -> Result<(), Error> {
    println!("Hello, world!");
    let mut events_loop = EventsLoop::new();
    let window = Window::new(&events_loop).unwrap();
    let hidpi_factor = window.get_hidpi_factor();
    let size = window.get_inner_size().unwrap().to_physical(hidpi_factor);
    let instance = wgpu::Instance::new();
    let surface = instance.create_surface(&window);
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });
    let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    fn compile_shadercode(code: &str, stage: glsl_to_spirv::ShaderType) -> Vec<u32> {
        use std::io::Read;

        let mut spv_bytes = Vec::new();
        glsl_to_spirv::compile(&code, stage).unwrap().read_to_end(&mut spv_bytes).unwrap();
        let mut spv_words = Vec::new();
        for bytes4 in spv_bytes.chunks(4) {
            spv_words.push(u32::from_le_bytes([bytes4[0], bytes4[1], bytes4[2], bytes4[3]]));
        }
        spv_words
    }

    let vs_words = compile_shadercode(r#"
    #version 450
    layout(location = 0) out vec2 v_TexCoord;
    layout(set = 1, binding = 0) uniform WorldPosScale {
        float world_x;
        float world_y;
        float world_sx;
        float world_sy;
    };
    layout(set = 2, binding = 0) uniform PosTex {
        float x;
        float y;
        float sx;
        float sy;
        float tx;
        float ty;
        float stx;
        float sty;
    };
    // redundant definition?
    //out gl_PerVertex {
        //vec4 gl_Position;
    //};
    const vec2 positions[6] = vec2[6](
        vec2( 0.0,  0.0),
        vec2( 1.0,  1.0),
        vec2( 0.0,  1.0),

        vec2( 0.0,  0.0),
        vec2( 1.0,  0.0),
        vec2( 1.0,  1.0)
    );
    const vec2 tex[6] = vec2[6](
        vec2(1.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),

        vec2(1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(0.0, 0.0)
    );
    void main() {
        gl_Position = vec4((positions[gl_VertexIndex].x*sx + x + world_x) * world_sx, (positions[gl_VertexIndex].y*sy + y + world_y) * world_sy, 0.0, 1.0);
        v_TexCoord = vec2(tex[gl_VertexIndex].x*stx + tx, tex[gl_VertexIndex].y*sty + ty);
    }"#, glsl_to_spirv::ShaderType::Vertex);
    let vs_module = device.create_shader_module(&vs_words);
    let fs_words = compile_shadercode(r#"
    #version 450
    layout(location = 0) in vec2 v_TexCoord;
    layout(location = 0) out vec4 outColor;
    layout(set = 0, binding = 0) uniform texture2D t_Color;
    layout(set = 0, binding = 1) uniform sampler s_Color;
    void main() {
        vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
        float mag = length(v_TexCoord-vec2(0.5));
        outColor = mix(tex, vec4(0.0), mag*mag);
    }"#, glsl_to_spirv::ShaderType::Fragment);
    let fs_module = device.create_shader_module(&fs_words);

    let img = open("./resources/littleroot.png").unwrap().to_rgba();
    let (twidth, theight) = img.dimensions();
    let texels: Vec<u8> = (0..theight).rev().flat_map(|y| (0..twidth).rev().map(move |x| (x,y)))
                            .flat_map(|(x,y)|
                                img.get_pixel(x,y).channels().into_iter().cloned()
                            ).collect();
    let texture_extant = wgpu::Extent3d {
        width: twidth,
        height: theight,
        depth: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extant,
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::TRANSFER_DST,
    });
    let texture_view = texture.create_default_view();
    let temp_buf = device.create_buffer_mapped(texels.len(), wgpu::BufferUsage::TRANSFER_SRC)
                         .fill_from_slice(&texels);
    let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    init_encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &temp_buf,
            offset: 0,
            row_pitch: 4*twidth,
            image_height: theight,
        },
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            array_layer: 0,
            origin: wgpu::Origin3d {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        texture_extant,
    );
    let init_command_buf = init_encoder.finish();
    device.get_queue().submit(&[init_command_buf]);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp:  100.0,
        compare_function: wgpu::CompareFunction::Always,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture,
            },
            wgpu::BindGroupLayoutBinding {
                binding: 1,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let world_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer,
            },
        ],
    });
    let local_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer,
            },
        ],
    });
    let mut world = World::new(0.0, 0.0, 1.0, &device, &world_bind_group_layout);
    let mut squares = vec![
        Square::new(0.0, 0.0, twidth as f32, theight as f32, 0.0, 0.0, 1.0, 1.0, &device, &local_bind_group_layout),
        Square::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, &device, &local_bind_group_layout),
    ];
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout, &world_bind_group_layout, &local_bind_group_layout],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::PipelineStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::PipelineStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        },
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format:      wgpu::TextureFormat::Bgra8UnormSrgb,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask:  wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[],
        sample_count: 1,
    });

    let mut swap_chain_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width.round() as u32,
        height: size.height.round() as u32,
        present_mode: wgpu::PresentMode::Vsync,
    };
    let mut swap_chain = device.create_swap_chain( &surface, &swap_chain_descriptor);
    let mut running = true;
    world.set_scale(1.0/(theight as f32));
    let mut save = if let Ok(file) = File::open("./save.json") {
        serde_json::from_reader(file)?
    } else {
        Save {
            x_increment: 1.0,
            y_increment: 1.0,
            x_start: 0.0,
            y_start: 0.0,
            width_in_tiles:  0,
            height_in_tiles: 0,
            hash_to_type: BTreeMap::new(),
        }
    };
    let to_tile_x = |x, save: &Save| ((x - save.x_start) / save.x_increment) as usize;
    let to_tile_y = |y, save: &Save| ((y - save.y_start) / save.y_increment) as usize;
    let hash_tile = |x, y, save: &Save| {
        let mut hasher = DefaultHasher::new();
        for (px, py) in (0..(save.x_increment as u32)).flat_map(|px| (0..(save.y_increment as u32)).map(move |py| (px,py))) {
            let global_x = px + save.x_start as u32 + save.x_increment as u32 * x as u32;
            let global_y = py + save.y_start as u32 + save.y_increment as u32 * y as u32;
            let double_reversed_index = ((theight-global_y)*twidth + (twidth-global_x)) * 4;
            for pixel in double_reversed_index..(double_reversed_index+4) {
                texels[pixel as usize].hash(&mut hasher);
            }
        }
        hasher.finish()
    };
    let mut refresh_squares = |squares: &mut Vec<Square>, save: &Save, device: &wgpu::Device, local_bind_group_layout: &wgpu::BindGroupLayout| {
        squares.drain(2..);
        for i in 0..save.width_in_tiles {
            for j in 0..save.height_in_tiles {
                let size = save.hash_to_type.get(&hash_tile(i,j, &save)).cloned().unwrap_or(0.5f32);
                squares.push(Square::new(
                save.x_start + (i as f32 + (1.0f32-size)/2.0) * save.x_increment,
                save.y_start + (j as f32 + (1.0f32-size)/2.0) * save.y_increment,
                save.x_increment*size,
                save.y_increment*size,
                0.0, 0.0, 1.0, 1.0, device, local_bind_group_layout));
            }
        }
    };
    squares[1].set_pos(save.x_start, save.y_start);
    squares[1].set_siz(save.x_increment, save.y_increment);
    let mut collide = false;
    while running {
        events_loop.poll_events(|event| {
            println!("{:?} event is", event);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    // raw_code is fallback if virtual_keycode is None because of bug in winit (I think)
                    WindowEvent::KeyboardInput { input: KeyboardInput { scancode: raw_code,
                                                                        virtual_keycode: maybe_virt_code,
                                                                        state: ElementState::Pressed,
                                                                        ..
                                                                      },
                                                  ..
                    } => match (raw_code, maybe_virt_code) {
                        (1,  None)  | (_, Some(VirtualKeyCode::Escape)) => running = false,
                        (44, None)  | (_, Some(VirtualKeyCode::Z))  => {
                            println!("\nsetting x/y increment {},{} !{},{}!\n", squares[1].x, squares[1].y, squares[1].sx, squares[1].sy);
                            save.x_increment = squares[1].sx;
                            save.y_increment = squares[1].sy;
                        },
                        (45, None)  | (_, Some(VirtualKeyCode::X))  => {
                            println!("\nsetting x/y start !{},{}! {},{}\n", squares[1].x, squares[1].y, squares[1].sx, squares[1].sy);
                            save.x_start = squares[1].x;
                            save.y_start = squares[1].y;
                        },
                        (46, None)  | (_, Some(VirtualKeyCode::C))  => {
                            save.width_in_tiles =  to_tile_x(squares[1].x, &save);
                            save.height_in_tiles = to_tile_y(squares[1].y, &save);
                            println!("\nsetting tile widht/height {},{} {},{} - !{}, {}!\n", squares[1].x, squares[1].y, squares[1].sx, squares[1].sy, save.width_in_tiles, save.height_in_tiles);
                        },
                        (47, None)  | (_, Some(VirtualKeyCode::V))  => {
                            serde_json::to_writer(File::create("./save.json").unwrap(), &save).unwrap();
                            println!("\nsaved!\n");
                        },
                        (57, None)  | (_, Some(VirtualKeyCode::Space))  => {
                            refresh_squares(&mut squares, &save, &device, &local_bind_group_layout);
                        },
                        (16, None)  | (_, Some(VirtualKeyCode::Q))  => {
                            save.hash_to_type.insert(hash_tile(to_tile_x(squares[1].x, &save), to_tile_y(squares[1].y, &save), &save), 0.9f32);
                            refresh_squares(&mut squares, &save, &device, &local_bind_group_layout);
                        },
                        (18, None)  | (_, Some(VirtualKeyCode::E))  => {
                            save.hash_to_type.insert(hash_tile(to_tile_x(squares[1].x, &save), to_tile_y(squares[1].y, &save), &save), 0.1f32);
                            refresh_squares(&mut squares, &save, &device, &local_bind_group_layout);
                        },
                        (19, None)  | (_, Some(VirtualKeyCode::R))  => {
                            collide = !collide;
                        },
                        // top left corner
                        (30, None)  | (_, Some(VirtualKeyCode::A))      => if !collide || save.hash_to_type.get(&hash_tile(to_tile_x(squares[1].x-save.x_increment, &save), to_tile_y(squares[1].y, &save), &save)).cloned().unwrap_or(1.0) < 0.5 { squares[1].pos_delta(-save.x_increment, 0.00f32) },
                        (32, None)  | (_, Some(VirtualKeyCode::D))      => if !collide || save.hash_to_type.get(&hash_tile(to_tile_x(squares[1].x+save.x_increment, &save), to_tile_y(squares[1].y, &save), &save)).cloned().unwrap_or(1.0) < 0.5 { squares[1].pos_delta( save.x_increment, 0.00f32) },
                        (17, None)  | (_, Some(VirtualKeyCode::W))      => if !collide || save.hash_to_type.get(&hash_tile(to_tile_x(squares[1].x, &save), to_tile_y(squares[1].y-save.y_increment, &save), &save)).cloned().unwrap_or(1.0) < 0.5 { squares[1].pos_delta( 0.00f32,-save.y_increment) },
                        (31, None)  | (_, Some(VirtualKeyCode::S))      => if !collide || save.hash_to_type.get(&hash_tile(to_tile_x(squares[1].x, &save), to_tile_y(squares[1].y+save.y_increment, &save), &save)).cloned().unwrap_or(1.0) < 0.5 { squares[1].pos_delta( 0.00f32, save.y_increment) },
                        // bottom right corner
                        (36, None)  | (_, Some(VirtualKeyCode::J))      => squares[1].siz_delta(-1.00f32, 0.00f32),
                        (38, None)  | (_, Some(VirtualKeyCode::L))      => squares[1].siz_delta( 1.00f32, 0.00f32),
                        (23, None)  | (_, Some(VirtualKeyCode::I))      => squares[1].siz_delta( 0.00f32,-1.00f32),
                        (37, None)  | (_, Some(VirtualKeyCode::K))      => squares[1].siz_delta( 0.00f32, 1.00f32),
                        // world translation
                        (105, None) | (_, Some(VirtualKeyCode::Left))   => world.pos_delta( save.x_increment, 0.00f32),
                        (106, None) | (_, Some(VirtualKeyCode::Right))  => world.pos_delta(-save.x_increment, 0.00f32),
                        (103, None) | (_, Some(VirtualKeyCode::Up))     => world.pos_delta( 0.00f32, save.y_increment),
                        (108, None) | (_, Some(VirtualKeyCode::Down))   => world.pos_delta( 0.00f32,-save.y_increment),
                        // world scale
                        (12, None)  | (_, Some(VirtualKeyCode::Minus))                                     => world.scale_down(),
                        (13, None)  | (_, Some(VirtualKeyCode::Add)) | (_, Some(VirtualKeyCode::Equals))   => world.scale_up(),
                        _ => println!("ignoring keycode {:?} / {:?}", raw_code, maybe_virt_code),
                    },
                    WindowEvent::Resized(size) => {
                        let physical = size.to_physical(hidpi_factor);
                        println!("resizing to {:?}", physical);
                        swap_chain_descriptor.width = physical.width.round() as u32;
                        swap_chain_descriptor.height = physical.height.round() as u32;
                        swap_chain = device.create_swap_chain( &surface, &swap_chain_descriptor);
                    },
                    WindowEvent::CloseRequested => running = false,
                    _ => {},
                },
                _ => {}
            }
        });

        let frame = swap_chain.get_next_texture();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo : 0 });
        world.maybe_update(&device, &mut encoder);
        for square in &mut squares {
            square.maybe_update(&device, &mut encoder);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::GREEN,
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&render_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_bind_group(1, &world.world_bind_group, &[]);
            for square in squares.iter() {
                rpass.set_bind_group(2, &square.local_bind_group, &[]);
                rpass.draw(0..6, 0..1);
            }
        }
        device.get_queue().submit(&[encoder.finish()]);
    }
    Ok(())
}
