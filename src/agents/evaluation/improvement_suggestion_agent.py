from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import Dict, Any, List
import json
import re
from config.options import LANGUAGE_OPTIONS

# Language-specific prompts for improvement suggestions
IMPROVEMENT_PROMPTS = {
    "en": {
        "prompt": """Respond exclusively in {language_name}. Analyze the evaluation findings and extract specific improvement instructions for each content section. Do NOT generate new content, only provide clear instructions on how to fix the identified problems.

PROPERTY DATA:
{property_data}

TARGET LANGUAGE: {language_name}
TARGET TONE: {tone}

EVALUATION RESULTS:
{scores_text}

CURRENT CONTENT:
{current_content}

IDENTIFIED ISSUES:
{issues_text}

TASK: Parse the evaluation findings and create specific instructions for fixing each identified problem. These instructions will be used by another LLM to correct the content. Base your instructions ONLY on the evaluation findings provided above.

IMPORTANT: For each section, you must include:
- "None" if there are NO problems identified for that specific section
- A detailed description with ALL problems and solutions found for that section ONLY

You MUST respond with a valid JSON object in this EXACT format:
{{
  "title": "None" OR "Problem: [state the specific issue found for TITLE]. Fix: [provide clear instruction on how to resolve it]",
  "meta_description": "None" OR "Problem: [state the specific issue found for META DESCRIPTION]. Fix: [provide clear instruction on how to resolve it]",
  "h1": "None" OR "Problem: [state the specific issue found for H1]. Fix: [provide clear instruction on how to resolve it]",
  "description": "None" OR "Problem: [state the specific issue found for DESCRIPTION]. Fix: [provide clear instruction on how to resolve it]",
  "key_features": "None" OR "Problem: [state the specific issue found for KEY FEATURES]. Fix: [provide clear instruction on how to resolve it]",
  "neighborhood": "None" OR "Problem: [state the specific issue found for NEIGHBORHOOD]. Fix: [provide clear instruction on how to resolve it]",
  "call_to_action": "None" OR "Problem: [state the specific issue found for CALL TO ACTION]. Fix: [provide clear instruction on how to resolve it]"
}}

CRITICAL: Only include problems that specifically belong to each section. Do not mix problems from different sections. Use "None" for sections with no identified issues.""",
    },
    "es": {
        "prompt": """Responde exclusivamente en {language_name}. Analiza los hallazgos de evaluación y extrae instrucciones específicas de mejora para cada sección de contenido. NO generes contenido nuevo, solo proporciona instrucciones claras sobre cómo solucionar los problemas identificados.

DATOS DE LA PROPIEDAD:
{property_data}

IDIOMA OBJETIVO: {language_name}
TONO OBJETIVO: {tone}

RESULTADOS DE EVALUACIÓN:
{scores_text}

CONTENIDO ACTUAL:
{current_content}

PROBLEMAS IDENTIFICADOS:
{issues_text}

TAREA: Parsea los hallazgos de evaluación y crea instrucciones específicas para corregir cada problema identificado. Estas instrucciones serán utilizadas por otro LLM para corregir el contenido. Basa tus instrucciones ÚNICAMENTE en los hallazgos de evaluación proporcionados arriba.

IMPORTANTE: Para cada sección, debes incluir:
- "None" si NO hay problemas identificados para esa sección específica
- Una descripción detallada con TODOS los problemas y soluciones encontrados para esa sección ÚNICAMENTE

DEBES responder con un objeto JSON válido en este formato EXACTO:
{{
  "title": "None" O "Problema: [indica el problema específico encontrado para TÍTULO]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "meta_description": "None" O "Problema: [indica el problema específico encontrado para META DESCRIPCIÓN]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "h1": "None" O "Problema: [indica el problema específico encontrado para H1]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "description": "None" O "Problema: [indica el problema específico encontrado para DESCRIPCIÓN]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "key_features": "None" O "Problema: [indica el problema específico encontrado para CARACTERÍSTICAS CLAVE]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "neighborhood": "None" O "Problema: [indica el problema específico encontrado para VECINDARIO]. Solución: [proporciona instrucción clara sobre cómo resolverlo]",
  "call_to_action": "None" O "Problema: [indica el problema específico encontrado para LLAMADA A LA ACCIÓN]. Solución: [proporciona instrucción clara sobre cómo resolverlo]"
}}

CRÍTICO: Solo incluye problemas que específicamente pertenezcan a cada sección. No mezcles problemas de diferentes secciones. Usa "None" para secciones sin problemas identificados.""",
    },
    "pt": {
        "prompt": """Responda exclusivamente em {language_name}. Analise os achados de avaliação e extraia instruções específicas de melhoria para cada seção de conteúdo. NÃO gere conteúdo novo, apenas forneça instruções claras sobre como resolver os problemas identificados.

DADOS DA PROPRIEDADE:
{property_data}

IDIOMA ALVO: {language_name}
TOM ALVO: {tone}

RESULTADOS DA AVALIAÇÃO:
{scores_text}

CONTEÚDO ATUAL:
{current_content}

PROBLEMAS IDENTIFICADOS:
{issues_text}

TAREFA: Parse os achados de avaliação e crie instruções específicas para corrigir cada problema identificado. Essas instruções serão usadas por outro LLM para corrigir o conteúdo. Base suas instruções APENAS nos achados de avaliação fornecidos acima.

IMPORTANTE: Para cada seção, você deve incluir:
- "None" se NÃO há problemas identificados para essa seção específica
- Uma descrição detalhada com TODOS os problemas e soluções encontrados para essa seção APENAS

Você DEVE responder com um objeto JSON válido neste formato EXATO:
{{
  "title": "None" OU "Problema: [indique o problema específico encontrado para TÍTULO]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "meta_description": "None" OU "Problema: [indique o problema específico encontrado para META DESCRIÇÃO]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "h1": "None" OU "Problema: [indique o problema específico encontrado para H1]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "description": "None" OU "Problema: [indique o problema específico encontrado para DESCRIÇÃO]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "key_features": "None" OU "Problema: [indique o problema específico encontrado para CARACTERÍSTICAS CHAVE]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "neighborhood": "None" OU "Problema: [indique o problema específico encontrado para VIZINHANÇA]. Solução: [forneça instrução clara sobre como resolvê-lo]",
  "call_to_action": "None" OU "Problema: [indique o problema específico encontrado para CHAMADA PARA AÇÃO]. Solução: [forneça instrução clara sobre como resolvê-lo]"
}}

CRÍTICO: Inclua apenas problemas que especificamente pertencem a cada seção. Não misture problemas de diferentes seções. Use "None" para seções sem problemas identificados.""",
    },
}


class ImprovementSuggestionAgent(AssistantAgent):
    """
    Agent that takes evaluation results and provides specific improvement instructions
    for each content section of the HTML. The instructions are designed to be used
    by another LLM to fix the content issues.
    """

    def __init__(
        self,
        name="improvement_suggestion_agent",
        model="gemma3n:e2b",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": True,  # Enable JSON output
            "family": "unknown",
            "structured_output": True,
        },
    ):
        model_client = OllamaChatCompletionClient(model=model, model_info=model_info)
        super().__init__(
            name=name,
            model_client=model_client,
            system_message="You are a multilingual content analysis expert for real estate listings. Given evaluation results, extract and parse the specific problems found and provide clear fix instructions for each content section based solely on evaluation findings. Always respond in the target language specified. Focus on addressing only the issues identified in the evaluation results. Do not generate new content, only provide instructions.",
        )

    async def generate_section_improvements(
        self,
        current_content: str,
        evaluation_results: Dict[str, Any],
        property_data: Dict[str, Any],
        language: str = "en",
        tone: str = "professional",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate specific improvement instructions for each HTML content section.

        Args:
            current_content: Current HTML content
            evaluation_results: Results from CompleteEvaluator
            property_data: Property data for context
            language: Target language code (en, es, pt)
            tone: Target tone

        Returns:
            Dict with section-specific fix instructions
        """

        # Build comprehensive improvement prompt
        improvement_prompt = self._build_improvement_prompt(
            current_content=current_content,
            evaluation_results=evaluation_results,
            property_data=property_data,
            language=language,
            tone=tone,
        )

        # Get AI-generated instructions
        response = await self.run(task=improvement_prompt)
        suggestions_text = response.messages[-1].content.strip()

        # Parse and structure the instructions
        structured_suggestions = self._parse_suggestions_response(
            suggestions_text=suggestions_text, evaluation_results=evaluation_results
        )

        return structured_suggestions

    def _build_improvement_prompt(
        self,
        current_content: str,
        evaluation_results: Dict[str, Any],
        property_data: Dict[str, Any],
        language: str,
        tone: str,
    ) -> str:
        """Build comprehensive prompt for improvement instructions."""

        # Get language name from language code
        language_name = LANGUAGE_OPTIONS.get(language, {}).get("name", language)

        # Format scores
        seo_score = evaluation_results.get("seo", {}).get("score", 0.0)
        language_score = evaluation_results.get("language_match", {}).get("score", 0.0)
        tone_score = evaluation_results.get("tone_match", {}).get("score", 0.0)
        scores_text = f"SEO: {seo_score:.2f}, Language: {language_score:.2f}, Tone: {tone_score:.2f}\n"

        # Format issues
        all_issues = evaluation_results.get("all_findings", [])
        issues_text = "\n".join([f"  - {issue}" for issue in all_issues])

        # Get language-specific prompt template
        prompt_template = IMPROVEMENT_PROMPTS.get(language, IMPROVEMENT_PROMPTS["en"])["prompt"]
        print(
            prompt_template.format(
                language_name=language_name,
                property_data=json.dumps(obj=property_data, indent=2),
                tone=tone,
                scores_text=scores_text,
                current_content=current_content,
                issues_text=issues_text,
            )
        )

        return prompt_template.format(
            language_name=language_name,
            property_data=json.dumps(obj=property_data, indent=2),
            tone=tone,
            scores_text=scores_text,
            current_content=current_content,
            issues_text=issues_text,
        )

    def _parse_suggestions_response(
        self, suggestions_text: str, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse AI response into structured improvement instructions."""

        print("Raw suggestions response:", suggestions_text)
        suggestions = {}

        try:
            # Try to extract JSON from the response
            # Look for JSON object in the text (handle cases where there might be extra text)

            # Find JSON object in the response
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", suggestions_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                print("Extracted JSON:", json_str)

                # Parse the JSON
                suggestions_json = json.loads(json_str)

                # Convert to our expected format
                for section_key, instruction_text in suggestions_json.items():
                    if instruction_text and instruction_text.strip():  # Only include non-empty instructions
                        # Check if it's "None" or actual instruction
                        instruction_clean = instruction_text.strip()
                        if instruction_clean.lower() == "none":
                            suggestions[section_key] = {"suggestion": "None"}
                        else:
                            suggestions[section_key] = {"suggestion": instruction_clean}
            else:
                print("No JSON found, attempting fallback parsing...")
                # Fallback to the old parsing method if JSON is not found
                suggestions = self._parse_suggestions_fallback(suggestions_text, evaluation_results)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print("Attempting fallback parsing...")
            # Fallback to the old parsing method
            suggestions = self._parse_suggestions_fallback(suggestions_text, evaluation_results)
        except Exception as e:
            print(f"Unexpected error during parsing: {e}")
            print("Attempting fallback parsing...")
            suggestions = self._parse_suggestions_fallback(suggestions_text, evaluation_results)

        print("Final parsed suggestions:", suggestions)
        return suggestions

    def _parse_suggestions_fallback(
        self, suggestions_text: str, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Fallback parsing method for non-JSON responses."""

        suggestions = {}
        current_section = None
        current_suggestion = {}

        lines = suggestions_text.split(sep="\n")

        # Keywords for different languages
        section_keywords = ["SECTION:", "SECCIÓN:", "SEÇÃO:"]
        suggestions_keywords = ["Problem:", "Problema:", "Fix:", "Solución:", "Solução:"]

        def starts_with_any(line: str, keywords: list) -> str:
            """Check if line starts with any of the keywords and return the keyword."""
            for keyword in keywords:
                if line.startswith(keyword):
                    return keyword
            return ""

        for line in lines:
            line = line.strip()

            # Check for section start
            section_keyword = starts_with_any(line, section_keywords)
            if section_keyword:
                # Save previous section if exists
                if current_section and current_suggestion:
                    suggestions[current_section] = current_suggestion

                # Start new section
                section_name = line.replace(section_keyword, "").strip().lower()
                # Normalize section names from different languages
                section_mapping = {
                    "título": "title",
                    "title": "title",
                    "meta descripción": "meta_description",
                    "meta description": "meta_description",
                    "meta descrição": "meta_description",
                    "h1": "h1",
                    "descripción": "description",
                    "description": "description",
                    "descrição": "description",
                    "características clave": "key_features",
                    "key features": "key_features",
                    "características chave": "key_features",
                    "vecindario": "neighborhood",
                    "neighborhood": "neighborhood",
                    "vizinhança": "neighborhood",
                    "llamada a la acción": "call_to_action",
                    "call to action": "call_to_action",
                    "chamada para ação": "call_to_action",
                }
                current_section = section_mapping.get(section_name, section_name.replace(" ", "_"))
                current_suggestion = {
                    "suggestion": "",
                }
                continue

            # Check for suggestions/problems
            suggestion_keyword = starts_with_any(line, suggestions_keywords)
            if suggestion_keyword and current_section:
                if current_suggestion["suggestion"]:
                    current_suggestion["suggestion"] += " " + line
                else:
                    current_suggestion["suggestion"] = line
                continue

        # Save last section
        if current_section and current_suggestion:
            suggestions[current_section] = current_suggestion

        return suggestions
