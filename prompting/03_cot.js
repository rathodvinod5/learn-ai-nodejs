import { OpenAI } from "openai";

const client = new OpenAI({
  apiKey: "",
});

const SYSTEM_PROMPT = `You are an expert AI Engineer, You have to analyse the users input carefully and 
then you need to breakdown the problems into multiple sub problems before coming to an final result. Always breakdown 
the users intention and how to solve that problem and then step by step solve it.

We are going to follow the pipeline of "INITIAL", "THINK", "ANALYSE" and "OUTPUT" pipeline.
The pipeplin:
- "INITIAL" when the user gives an input we will have an initial thought process on what this user is trying to do.
- "THINK" this is where we are going to think about how to solve this and start to breakdown the problem.
- "ANALYSE" this is where we analyse the solution and also verify the output is correct.
- "THINK" we can go back to think mode where we now see if any sub problem remains and think.
- "ANALYSE" again analyse the problem and get on to a solution.
- "OUTPUT" this is where we can end and give the final output to the user.

Rules: 
- Always output one step at a time and wait for other steps for proceeding.
- Always maintain sequence of pipeline as given in the examples.
- Always follow JSON output format strictly.

Example:
- "USER": what is 2 + 2 - 5 * 10 / 3?
OUTPUT: 
- "INITIAL": "The user wants me to solve a mathematical equation"
- "THINK": I will use the BODMAS rule and based on that I should first multiply 5 * 10 which is 50"
- "ANALYSE": "Yes, the bodmas is actually right and now equation is 2 + 2 - 50 / 3"
- "THINK": "Now as per the rule I should perform division operation, 50 / 3 which is which is 16.666667" 
- "ANALYSE": "Now the new equations remains 2 + 2 - 16.666667"
- "THINK": "Now it is simple we can just do 2 + 2 = 4 and the equation remains is 4 - 16.666667"
- "ANALYSE": "Great, now lets just do the final step as simple subtraction"
- "THINK": "After the final subtraction the answer remains: -12.666667"
- "OUTPUT": "The final output is: -12.666667" 

Output format:
{
    "step": "INITIAL" | "THINK" | "ANALYSE" | "OUTPUT",
    "text": "<Actual text>"
}
`;

const MESSAGES_DB = [
  {
    role: "system",
    content: SYSTEM_PROMPT,
  },
];

async function main(prompt = "") {
  MESSAGES_DB.push({ role: "user", content: prompt });
  console.log("\n==================================================");
  while (true) {
    const result = await client.chat.completions.create({
      model: "gpt-4o",
      messages: MESSAGES_DB,
    });

    const rawData = result.choices[0].message.content;
    const cleanData = rawData.replace(/^```json\s*|```$/g, "").trim();
    const parsedResult = await JSON.parse(cleanData);

    MESSAGES_DB.push({ role: "assistant", content: rawData });

    console.log(`(${parsedResult.step}): ${parsedResult.text}`);

    if (parsedResult.step == "OUTPUT") break;
  }
  console.log("==================================================\n");
}

main("What is 4 + 6 + 9 - 3 * 5");
// main("What is the meaning of life?");
