import os
import json
import re
import time
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIG ==========
MODEL = "claude-sonnet-4-5-20250929"
RUNS_PER_PROMPT = 3
OUTPUT_FILE = "experiment_results_v5.json"

# Optional: approximate pricing (for reference only)
INPUT_COST_PER_1K = 0.003
OUTPUT_COST_PER_1K = 0.015

if "ANTHROPIC_API_KEY" not in os.environ:
    raise EnvironmentError("Please set your ANTHROPIC_API_KEY environment variable.")

client = Anthropic()

# ========== PROMPTS ==========
PROMPTS_v4 = [
    {
        "name": "simple_summary",
        "prompt": (
            "Summarize the following text into a short title and a concise summary:\n\n"
            "'The quick brown fox jumps over the lazy dog, but this time it lands near a small stream that flows gently "
            "through a quiet forest. The sun rises over the trees, casting golden rays across the water, while birds sing "
            "softly in the distance. Watching the scene, the fox pauses to drink, reflecting on its morning adventure. "
            "Nearby, the lazy dog slowly wakes up, stretches, and yawns, realizing it had missed the action once again. "
            "Despite their differences, the fox and the dog share the peacefulness of a calm morning in nature.'"
        ),
        "schema": {"title": "string", "summary": "string"}
    },
    {
        "name": "medium_entities",
        "prompt": (
            "Extract all named entities and their types from this paragraph:\n\n"
            "'Barack Hussein Obama II, born on August 4, 1961, in Honolulu, Hawaii, is an American politician, author, "
            "and attorney who served as the 44th President of the United States from January 20, 2009, to January 20, 2017. "
            "A member of the Democratic Party, Obama succeeded George W. Bush and was followed by Donald J. Trump. "
            "Before becoming president, he represented Illinois in the U.S. Senate from 2005 to 2008, where he worked on issues "
            "related to nuclear non-proliferation, government transparency, and veterans' benefits. "
            "He previously served in the Illinois State Senate from 1997 to 2004. "
            "Obama graduated from Columbia University in New York City in 1983 with a degree in political science, "
            "specializing in international relations, and later earned his Juris Doctor from Harvard Law School, "
            "where he became the first African-American president of the Harvard Law Review. "
            "During his presidency, Obama enacted major policies including the Affordable Care Act, the Dodd–Frank Wall Street "
            "Reform Act, and the Deferred Action for Childhood Arrivals (DACA) program. "
            "In foreign policy, his administration oversaw the military operation that led to the death of al-Qaeda leader "
            "Osama bin Laden in Abbottabad, Pakistan, in 2011, and negotiated the Paris Climate Agreement in 2015. "
            "In recognition of his diplomatic efforts, he was awarded the Nobel Peace Prize in 2009. "
            "Obama is married to Michelle LaVaughn Robinson Obama, a lawyer, author, and former First Lady, "
            "known for her initiatives such as 'Let’s Move!' and 'Reach Higher'. The couple has two daughters, "
            "Malia Ann Obama and Natasha (Sasha) Obama. After leaving the White House, Barack and Michelle founded "
            "Higher Ground Productions and the Obama Foundation, both based in Chicago, Illinois. "
            "They continue to reside in Washington, D.C., and remain active in global education, leadership development, "
            "and civic engagement programs.'"
        ),
        "schema": {"entities": [{"type": "string", "value": "string"}]}
    },
    {
        "name": "complex_user_profile",
        "prompt": (
            "Generate a realistic user profile in JSON format with detailed demographics, preferences, and recent activities. "
            "Include details such as name, age, gender, country, city, profession, education level, income range, and personality traits. "
            "Under 'preferences', include favorite music genres, sports, movies, and technology interests. "
            "For 'recent_activities', include both online and offline events, such as social media engagement, travel, purchases, "
            "fitness activities, or learning achievements. Ensure timestamps are ISO 8601 formatted.\n\n"
            "Example context: The user is a tech-savvy professional living in a metropolitan area, active on social media, "
            "and interested in personal growth, AI tools, and outdoor fitness."
        ),
        "schema": {
            "user": {
                "name": "string",
                "age": "number",
                "gender": "string",
                "location": {"country": "string", "city": "string"},
                "profession": "string",
                "education": "string",
                "income_range": "string",
                "personality_traits": ["string"],
                "preferences": {
                    "music": ["string"],
                    "sports": ["string"],
                    "movies": ["string"],
                    "technology": ["string"]
                },
                "recent_activities": [{"activity": "string", "timestamp": "string"}]
            }
        }
    }
]

# ========== PROMPTS ==========
PROMPTS = [
    
    # ----------- NEW TRAP PROMPTS BELOW ------------
    {
        "name": "multi_blocks",
        "prompt": (
            "Read the following three short news headlines and produce a structured JSON output "
            "where each headline becomes a separate object in an array:\n\n"
            "1. 'Apple announces new AI-powered MacBook Pro.'\n"
            "2. 'NASA prepares for first crewed Mars mission.'\n"
            "3. 'Global markets rally after inflation cools.'\n\n"
            "For each headline, include fields: 'headline', 'category', and 'sentiment' "
            "(positive, neutral, or negative)."
        ),
        "schema": {"headlines": [{"headline": "string", "category": "string", "sentiment": "string"}]}
    },
    {
        "name": "escaped_quotes",
        "prompt": (
            "Output a JSON representation of the following conversation between two people, "
            "including who spoke and what they said. Include at least two sentences that contain quoted text.\n\n"
            "Person A: \"Did you hear what John said yesterday?\"\n"
            "Person B: \"Yes, he said, 'I'm joining the new AI project next month!'\"\n"
            "Person A: \"That's exciting! I might ask him more about it.\""
        ),
        "schema": {"conversation": [{"speaker": "string", "utterance": "string"}]}
    },
    {
        "name": "nested_deep",
        "prompt": (
            "Generate a detailed company structure for 'TechNova Industries' in JSON format. "
            "It should include departments, each with teams, each team with multiple employees. "
            "For each employee include name, role, skills, and years_of_experience."
        ),
        "schema": {
            "company": "string",
            "departments": [
                {
                    "name": "string",
                    "teams": [
                        {
                            "team_name": "string",
                            "members": [
                                {"name": "string", "role": "string", "skills": ["string"], "years_of_experience": "number"}
                            ]
                        }
                    ]
                }
            ]
        }
    },
    {
        "name": "long_text_field",
        "prompt": (
            "Create a JSON object representing an interview transcript. "
            "The transcript should include the speaker's name and a long response (at least 100 words) "
            "about the topic 'The future of AI in healthcare'."
        ),
        "schema": {"speaker": "string", "transcript": "string"}
    }
]


# ========== HELPERS ==========

def log(msg: str):
    """Prints timestamped progress messages."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def extract_json_from_xml_or_markdown(text: str) -> str:
    """Handles XML <response> tags and removes markdown ```json fences."""
    if not text:
        return text.strip()

    # --- Remove Markdown code fences ---
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I | re.M)
    text = re.sub(r"```$", "", text.strip(), flags=re.M)

    # --- Extract JSON inside XML-like tags ---
    match = re.search(r"<(?:response|json)>(.*?)</(?:response|json)>", text, re.S | re.I)
    if match:
        return match.group(1).strip()

    return text.strip()

def call_model(prompt_text: str):
    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt_text}]
    )
    text = response.content[0].text
    in_toks = response.usage.input_tokens
    out_toks = response.usage.output_tokens
    cost = (in_toks / 1000 * INPUT_COST_PER_1K) + (out_toks / 1000 * OUTPUT_COST_PER_1K)
    return text, in_toks, out_toks, cost

def try_parse_json(text: str):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

# ========== MAIN EXPERIMENT ==========

results = []
log("Starting experiment run...")
start_time = time.time()

for test in PROMPTS:
    log(f"\n=== Testing prompt: {test['name']} ===")
    for method in ["plain_json", "xml_wrapped"]:
        log(f"--- Method: {method} ---")

        successes = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        detailed_runs = []

        for run in range(RUNS_PER_PROMPT):
            schema_str = json.dumps(test["schema"], indent=2)
            if method == "plain_json":
                prompt_text = (
                    f"You must return a valid JSON strictly following this schema:\n{schema_str}\n\n"
                    f"Prompt:\n{test['prompt']}"
                )
            else:
                prompt_text = (
                    f"Wrap your JSON output inside <response>...</response> tags.\n"
                    f"Ensure the JSON strictly follows this schema:\n{schema_str}\n\n"
                    f"Prompt:\n{test['prompt']}"
                )

            log(f"Run {run + 1}/{RUNS_PER_PROMPT} for {test['name']} ({method})...")
            try:
                output, in_toks, out_toks, cost = call_model(prompt_text)
                json_text = extract_json_from_xml_or_markdown(output)
                parsed_ok = try_parse_json(json_text)
                if parsed_ok:
                    successes += 1
                    log("✅ JSON parsed successfully.")
                else:
                    log("⚠️ JSON parse failed.")

                detailed_runs.append({
                    "run": run + 1,
                    "raw_output": output,
                    "json_extracted": json_text,
                    "parsed_ok": parsed_ok,
                    "input_tokens": in_toks,
                    "output_tokens": out_toks,
                    "cost": cost
                })

                total_input_tokens += in_toks
                total_output_tokens += out_toks
                total_cost += cost
            except Exception as e:
                log(f"❌ Error: {e}")
                detailed_runs.append({"run": run + 1, "error": str(e)})

            time.sleep(1)

        log(f"Completed {method} for {test['name']} | Successes: {successes}/{RUNS_PER_PROMPT}\n")

        results.append({
            "prompt_name": test["name"],
            "method": method,
            "successes": successes,
            "total_runs": RUNS_PER_PROMPT,
            "avg_input_tokens": round(total_input_tokens / RUNS_PER_PROMPT, 1),
            "avg_output_tokens": round(total_output_tokens / RUNS_PER_PROMPT, 1),
            "avg_cost": round(total_cost / RUNS_PER_PROMPT, 6),
            "details": detailed_runs
        })

# ========== SAVE RESULTS ==========
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ========== PRINT SUMMARY ==========
log("\n=== EXPERIMENT SUMMARY ===")
for r in results:
    print(
        f"{r['prompt_name']:<20} | {r['method']:<12} | "
        f"Success: {r['successes']}/{r['total_runs']} | "
        f"Input: {r['avg_input_tokens']} | Output: {r['avg_output_tokens']} | "
        f"Cost(avg): ${r['avg_cost']}"
    )

elapsed = round(time.time() - start_time, 2)
log(f"\nExperiment completed in {elapsed} seconds.")
log(f"Detailed results saved to: {OUTPUT_FILE}")
