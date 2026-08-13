import { OpenAI } from "openai";

const client = new OpenAI({
  apiKey: "",
});

async function main() {
  const result = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "What is 5 + 5" }],
  });
  // .then((response) => {
  //   console.log(response.choices[0].message.content);
  // });
  console.log("Response from AI: ", result.choices[0].message.content);
}

main();
