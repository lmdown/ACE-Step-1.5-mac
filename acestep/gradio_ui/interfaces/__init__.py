"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import gradio as gr
from acestep.gradio_ui.i18n import get_i18n, t
from acestep.gradio_ui.interfaces.dataset import create_dataset_section
from acestep.gradio_ui.interfaces.generation import create_generation_section
from acestep.gradio_ui.interfaces.result import create_results_section
from acestep.gradio_ui.interfaces.training import create_training_section
from acestep.gradio_ui.events import setup_event_handlers, setup_training_event_handlers
from acestep.gradio_ui.events import generation_handlers as gen_h


def create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=None, language='en') -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        dataset_handler: Dataset handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja', default: 'en')
        
    Returns:
        Gradio Blocks instance
    """
    # Initialize i18n with selected language
    i18n = get_i18n(language)
    
    with gr.Blocks(
        title=t("app.title"),
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .lm-hints-row {
            align-items: stretch;
        }
        .lm-hints-col {
            display: flex;
        }
        .lm-hints-col > div {
            flex: 1;
            display: flex;
        }
        .lm-hints-btn button {
            height: 100%;
            width: 100%;
        }
        /* Position Audio time labels lower to avoid scrollbar overlap */
        .component-wrapper > .timestamps {
            transform: translateY(15px);
        }
        """,
    ) as demo:
        
        gr.HTML(f"""
        <div class="main-header">
            <h1>{t("app.title")}</h1>
            <p>{t("app.subtitle")}</p>
        </div>
        """)
        
        # Dataset Explorer Section
        dataset_section = create_dataset_section(dataset_handler)
        
        # Generation Section (pass init_params and language to support pre-initialization)
        generation_section = create_generation_section(dit_handler, llm_handler, init_params=init_params, language=language)
        
        # Results Section
        results_section = create_results_section(dit_handler)
        
        # Training Section (LoRA training and dataset builder)
        # Pass init_params to support hiding in service mode
        training_section = create_training_section(dit_handler, llm_handler, init_params=init_params)
        
        # Connect event handlers
        setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section)
        
        # Connect training event handlers
        setup_training_event_handlers(demo, dit_handler, llm_handler, training_section)
        
        # Add automatic initialization for base model on startup
        service_pre_initialized = init_params is not None and init_params.get('pre_initialized', False)
        if not service_pre_initialized:
            # Check if default model is base - only auto-init if config_path specifies base model
            available_models = dit_handler.get_available_acestep_v15_models()
            config_path_value = init_params.get('config_path', '') if init_params else ''
            if "acestep-v15-base" in available_models and "acestep-v15-base" in config_path_value:
                # Trigger auto-initialization when demo loads
                demo.load(
                    # First update status to show initialization is starting
                    fn=lambda: gr.update(value="Initializing base model..."),
                    outputs=[generation_section["init_status"]]
                ).then(
                    # Simulate init_btn click with base model parameters
                    fn=lambda *args: gen_h.init_service_wrapper(dit_handler, llm_handler, *args),
                    inputs=[
                        generation_section["checkpoint_dropdown"],
                        generation_section["config_path"],
                        generation_section["device"],
                        generation_section["init_llm_checkbox"],
                        generation_section["lm_model_path"],
                        generation_section["backend_dropdown"],
                        generation_section["use_flash_attention_checkbox"],
                        generation_section["offload_to_cpu_checkbox"],
                        generation_section["offload_dit_to_cpu_checkbox"],
                        generation_section["compile_model_checkbox"],
                        generation_section["quantization_checkbox"],
                    ],
                    outputs=[
                        generation_section["init_status"], 
                        generation_section["generate_btn"], 
                        generation_section["service_config_accordion"],
                        # Model type settings (updated based on actual loaded model)
                        generation_section["inference_steps"],
                        generation_section["guidance_scale"],
                        generation_section["use_adg"],
                        generation_section["shift"],
                        generation_section["cfg_interval_start"],
                        generation_section["cfg_interval_end"],
                        generation_section["task_type"],
                    ]
                )
    
    return demo
