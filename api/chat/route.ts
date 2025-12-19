import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const body = await req.json();
  const message = body.message;

  if (!message) {
    return NextResponse.json(
      { error: 'Message is required' },
      { status: 400 }
    );
  }

  // Call your existing AI logic here
  const reply = await runYourExistingAgent(message);

  return NextResponse.json({ reply });
}
