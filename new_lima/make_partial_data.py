import json
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

client = OpenAI()

def get_prompt(doc):
  prompt = """Please generate an instruction-following response that meets the _partial_ specification provided below.
  The response is free to be about anything as long as it meets the description/criterion provided in the partial specification.
  Make sure it's a high-quality response to some, any, hypothetical request that fits with the partial specification.
  Don't reference this task in your response, as that would make it a bad response for the hypothetical request.
  For example, do not say "Certainly! Here’s a high-quality response to a hypothetical request..."

  Partial specification:  {}
  """.format(doc)
  return prompt

def fetch_response(prompt):
    # Replace 'your_api_key_here' with your actual OpenAI API key
    #openai.api_key = 'your_api_key_here'
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages =[{'role': 'user', 'content': prompt}],
        max_tokens=1500
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

constraints = [ "exactly 5 words", "code with absolutely no commentary", "only in the past tense", "no more than 3 sentences", "must include a question", "beginning with a specific word", "without using the letter 'e'", "only positive statements", "include at least one number", "no longer than 50 characters", "using only simple sentences", "limited to 100 words", "in the form of a question", "containing at least one quote", "must rhyme", "using bullet points", "in alphabetical order", "without repeating any words", "written as a poem", "using only active voice", "short recipe", "step-by-step guide", "checklist", "bullet points", "FAQ", "summary", "example scenario", "pros and cons list", "troubleshooting steps", "comparison table", "quick tips", "tutorial", "case study", "interactive quiz", "story or anecdote", "workflow", "annotated diagram", "glossary", "timeline", "personalized recommendation", "decision tree", "example dialogue", "template", "flowchart", "overview", "interactive simulation", "predictive model", "ranking list", "mind map", "visual infographic", "snippet"] 

length_constraints = ["{} words or less".format(x) for x in [1, 2, 3, 4, 5, 10, 20, 40, 80, 160, 360, 720]]

prog_langs = [ "Python", "JavaScript", "Java", "C", "C++", "C#", "Ruby", "Swift", "Go", "Kotlin", "R", "TypeScript", "PHP", "Perl", "Scala", "Rust", "Objective-C", "SQL", "MATLAB", "Dart", "Haskell"]
prog_langs = [x.lower() for x in prog_langs]

prog_langs = [x + " snippet" for x in prog_langs]

strategies = [ "Step-by-step", "Breakdown", "Examples", "Clarification", "Reframing", "Simplification", "Analogies", "Constraints", "Prioritization", "Verification", "Elaboration", "Assumptions", "Scenarios", "Iterative", "Visuals", "Checkpoints", "Decomposition", "Feedback", "Comparisons", "Focus" ]
strategies = ['answer using {}'.format(x.lower()) for x in strategies]


nat_langs = {
  "Arabic": "مرحبا بك في عالم الذكاء الاصطناعي. كيف يمكنني مساعدتك اليوم؟",
  "Bengali": "আপনাকে কৃত্রিম বুদ্ধিমত্তার জগতে স্বাগতম। আজকে আমি কিভাবে আপনাকে সাহায্য করতে পারি?",
  "Finnish": "Tervetuloa tekoälyn maailmaan. Kuinka voin auttaa sinua tänään?",
  "Indonesian": "Selamat datang di dunia kecerdasan buatan. Bagaimana saya bisa membantu Anda hari ini?",
  "Japanese": "人工知能の世界へようこそ。今日はどのようにお手伝いできますか？",
  "Kiswahili": "Karibu kwenye ulimwengu wa akili bandia. Ninaweza kukusaidia vipi leo?",
  "Korean": "인공지능의 세계에 오신 것을 환영합니다. 오늘 어떻게 도와드릴까요?",
  "Russian": "Добро пожаловать в мир искусственного интеллекта. Как я могу помочь вам сегодня?",
  "Telugu": "కృత్రిమ మేధస్సు ప్రపంచంలోకి స్వాగతం. నేను మీకు ఈ రోజు ఎలా సహాయం చేయగలను?",
  "Thai": "ยินดีต้อนรับสู่โลกของปัญญาประดิษฐ์ วันนี้ฉันสามารถช่วยคุณได้อย่างไร?"
}


with open('partial_spec2.jsonl', 'w') as fout:
  for lang in nat_langs:
    example = {'messages': [{"role": "user", "content": lang},
      {"role": "assistant", "content": nat_langs[lang]}]}
    fout.write(json.dumps(example) + '\n')
  #for constraint in tqdm(constraints + prog_langs + strategies + length_constraints):
  for constraint in tqdm(strategies):
    print(constraint)
    response = fetch_response(get_prompt(constraint))
    example = {'messages': [{"role": "user", "content": constraint},
      {"role": "assistant", "content": response}]}
    fout.write(json.dumps(example) + '\n')
