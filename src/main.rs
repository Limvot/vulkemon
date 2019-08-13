use wgpu::winit::*;
use image::*;

fn main() {
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
    // redundant definition?
    //out gl_PerVertex {
        //vec4 gl_Position;
    //};
    const vec2 positions[6] = vec2[6](
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5,  0.5),

        vec2(-0.5, -0.5),
        vec2( 0.5, -0.5),
        vec2( 0.5,  0.5)
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
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        v_TexCoord = tex[gl_VertexIndex];
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
        //outColor = vec4(1.0, 0.0, 0.0, 1.0);
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
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
    while running {
        events_loop.poll_events(|event| {
            println!("{:?} event is", event);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(code),
                                                                        state: ElementState::Pressed,
                                                                        ..
                                                                      },
                                                  ..
                    } => match code {
                        VirtualKeyCode::Escape => running = false,
                        _ => {
                            println!("ignoring keycode {:?}", code);
                        },
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
            rpass.draw(0..6, 0..1);
        }
        device.get_queue().submit(&[encoder.finish()]);
    }
}
