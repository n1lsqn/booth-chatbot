import { openai } from "@ai-sdk/openai";
import { DataAPIClient } from "@datastax/astra-db-ts";
import { streamText } from "ai";
import { configDotenv } from "dotenv";
import OpenAI from "openai";

configDotenv();

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  OPENAI_API_KEY
} = process.env;

console.log(ASTRA_DB_NAMESPACE, ASTRA_DB_COLLECTION, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, OPENAI_API_KEY);

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT!, { namespace: ASTRA_DB_NAMESPACE });
const openAi = new OpenAI({apiKey: OPENAI_API_KEY});

export async function POST(req: Request) {
  const { messages } = await req.json();
  const latestMessage = messages[messages.length - 1]?.content;

  let docContext = "";

  const embeddings = await openAi.embeddings.create({
    model: "text-embedding-3-small",
    input: latestMessage,
    encoding_format: "float"
  })

  const collection = await db.collection(ASTRA_DB_COLLECTION!);
  const cursor = collection.find(
    {},
    {
      sort: {
        $vector: embeddings.data[0].embedding
      },
      limit: 10
    }
  )

  const documents = await cursor.toArray();

  for await (const doc of documents) {
    docContext += doc.text + " ";
  }

  const template = {
    role: "system",
    content: `
      あなたはVRChatに詳しいです。
      コンテキストで受け取った情報をもとに、このショップについての質問に答えることができます。
      ----------
      ${docContext}
      ----------
      ----------
      Questions: ${latestMessage}
      ----------
      `,
  };
  
  const result = await streamText({
    model: openai("gpt-3.5-turbo"),
    prompt: template.content,
  })

  return result.toDataStreamResponse();
}