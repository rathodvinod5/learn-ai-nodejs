import { OpenAI } from "openai";
import axios from "axios";
import { exec } from "child_process";

const client = new OpenAI({
  apiKey: "",
});

const SYSTEM_PROMPT = `You are an expert AI Engineer, 

Perona: You are a senior software Engineer
Persona Traits:
- You always sound technical
- You never answer back on personal things and you don't have a personal life
- All you know is how and what code is

You have to analyse the users input carefully and 
then you need to breakdown the problems into multiple sub problems before coming to an final result. Always breakdown 
the users intention and how to solve that problem and then step by step solve it.

We are going to follow the pipeline of "INITIAL", "THINK", "TOOL_REQUEST" "ANALYSE" and "OUTPUT" pipeline.

The pipeplin:
- "INITIAL" when the user gives an input we will have an initial thought process on what this user is trying to do.
- "THINK" this is where we are going to think about how to solve this and start to breakdown the problem.
- "ANALYSE" this is where we analyse the solution and also verify the output is correct.
- "THINK" we can go back to think mode where we now see if any sub problem remains and think.
- "ANALYSE" again analyse the problem and get on to a solution.
- "TOOL_REQUEST" use this for calling or requesting a tool the format of the output would be
    { "step": "TOOL_REQUEST", "functionName": "getWeatherData", "input": "Goa"}
- "OUTPUT" this is where we can end and give the final output to the user.

Available tools:
- "getWatherData": getWatherData(cityName: string): Returns the realtime weather info the city.
- "executeCommandOnCli": executeCommandOnCli(cmd: string): Executes commands on users device and returns the output from stdout

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

Example:
- User: What is the weather of the Goa?
OUTPUT:
- "INITIAL": "The user wants me to fetch the weather of the Goa".
- "THINK": "Fromt he tools I can see we have a tool named getWeatherData which can be called",
- "ANALYSE": "We are going right, we can call getWeatherData with "GOA" as input"
- "TOOL_REQUEST": { functionName: "getWeatherData", input: "Goa" }
- "TOOL_OUTPUT": "The weather of the Goa is sunny with some 30 degree celcius"
- "THINK": "We got the weather info"
- "OUTPUT": "The weather of the Goa is sunny with some 30 degree celcius, it's gonna be hot"

Output format:
{
    "step": "INITIAL" | "THINK" | "TOOL_REQUEST" | "ANALYSE" | "OUTPUT",
    "text": "<Actual text>",
    "functionName": "<NAME of the function>",
    "input": "input params of the function"
}
`;

const MESSAGES_DB = [
  {
    role: "system",
    content: SYSTEM_PROMPT,
  },
];

async function getWeatherData(cityName = "") {
  // return `Weather of the ${cityName} is: 40 degree celcius `;
  const url = `https://wttr.in/${cityName.toLowerCase()}?format=%C+%t`;
  const response = await axios.get(url, { responseType: "text" });
  const result = JSON.stringify({ cityName, weatherInfo: response.data });
  return result;
}

async function executeCommandOnCli(cmd) {
  return new Promise((res, rej) => {
    exec(cmd, (err, out) => {
      if (err) return res("There was an error: " + err);
      else return res(out);
    });
  });
}

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

    if (parsedResult.step == "TOOL_REQUEST") {
      const { functionName, input } = parsedResult;

      switch (functionName) {
        case "executeCommandOnCli": {
          try {
            const toolResult = await executeCommandOnCli(input);
            console.log(`(${functionName}): ${input} - `, toolResult);

            MESSAGES_DB.push({
              role: "developer",
              content: JSON.stringify({
                step: "TOOL_OUTPUT",
                output: toolResult,
              }),
            });
          } catch (error) {
            MESSAGES_DB.push({
              role: "developer",
              content: JSON.stringify({
                status: "error",
                output: error,
              }),
            });
          }

          continue;
        }
        case "getWatherData":
          {
            try {
              const toolResult = await getWeatherData(input);
              console.log(`(${functionName}): ${input} - `, toolResult);

              MESSAGES_DB.push({
                role: "developer",
                content: JSON.stringify({
                  step: "TOOL_OUTPUT",
                  output: toolResult,
                }),
              });
            } catch (error) {
              MESSAGES_DB.push({
                role: "developer",
                content: JSON.stringify({
                  status: "error",
                  output: error,
                }),
              });
            }

            continue;
          }
          break;
      }
    }
  }
  console.log("==================================================\n");
}

// main("What is the weather of pune?");
// main("What is the weather of Pune, Mumbai and Bangalore?");
// main(
//   "What is the weather of Pune, Mumbai and Bangalore? and write the output to weather.txt file",
// );
// main(
//   `Hey create a small do to application with HTML, css and javascript that can be run in browser. Please keep all the files related
//   to todo application in a folder called as todo-app. After creating the application run in the browser please`,
// );
main("What is the meaning of life"); // if persona is enabled
