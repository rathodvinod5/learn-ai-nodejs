import { OpenAI } from "openai";

const client = new OpenAI({
  apiKey: "",
});

async function main() {
  const result = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "user",
        content: `What is 10 + 10?
        
        Do not add anything in answer, take the samples from the examples.

        Examples:
        - What is 2 + 4?
          Expected output: 6 (Six).
        - What is 2 + 1?
          Expected output: 3 (Three).
        `,
      },
    ],
  });

  console.log("Response from AI: ", result.choices[0].message.content);
}

main();
