import * as mindee from "mindee";
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const apiKey = process.env.MINDEE_API_KEY;
const modelId = process.env.MODEL_ID;

if (!apiKey || !modelId) {
    console.error('Missing required environment variables: MINDEE_API_KEY and/or MODEL_ID');
    process.exit(1);
}

// Init a new client
const mindeeClient = new mindee.ClientV2({ apiKey: apiKey });

async function processFile(inputPath: string): Promise<{rawText: string, structuredResult: string, fileName: string, processingTime: number}> {
    const startTime = Date.now();
    // Set inference parameters
    const inferenceParams = {
        modelId: modelId!,
        // Extract the full text content from the document as strings.
        rawText: true,
        // Options: set to `true` or `false` to override defaults
        // Enhance extraction accuracy with Retrieval-Augmented Generation.
        rag: undefined,
        // Calculate bounding box polygons for all fields.
        polygon: undefined,
        // Boost the precision and accuracy of all extractions.
        // Calculate confidence scores for all fields.
        confidence: undefined,
    };

    // Load a file from disk
    const inputSource = new mindee.PathInput({ inputPath: inputPath });

    // Send for processing using polling
    const response = await mindeeClient.enqueueAndGetInference(inputSource, inferenceParams);

    // Print a string summary of the structured output
    console.log(response.inference.toString());

    // Debug: Log the entire response structure to understand where raw text is located
    console.log('Full response structure:', JSON.stringify(response, null, 2));

    // Get full raw text as a single string
    let rawText = 'No raw text available';
    let pageTexts = '';

    // Try multiple approaches to extract raw text
    const responseAny = response as any;
    
    // Approach 1: Check if rawText is directly in the response
    if (responseAny.rawText) {
        console.log('Found rawText at response level');
        if (typeof responseAny.rawText === 'string') {
            rawText = responseAny.rawText;
        } else if (responseAny.rawText.content) {
            rawText = responseAny.rawText.content;
        } else if (responseAny.rawText.pages) {
            rawText = responseAny.rawText.pages.map((page: any) => page.content || page).join('\n\n');
        }
    }
    // Approach 2: Check if rawText is in inference
    else if (responseAny.inference && responseAny.inference.rawText) {
        console.log('Found rawText in inference');
        if (typeof responseAny.inference.rawText === 'string') {
            rawText = responseAny.inference.rawText;
        } else if (responseAny.inference.rawText.content) {
            rawText = responseAny.inference.rawText.content;
        } else if (responseAny.inference.rawText.pages) {
            rawText = responseAny.inference.rawText.pages.map((page: any) => page.content || page).join('\n\n');
        }
    }
    // Approach 3: Check if rawText is in inference.result
    else if (responseAny.inference && responseAny.inference.result && responseAny.inference.result.rawText) {
        console.log('Found rawText in inference.result');
        if (typeof responseAny.inference.result.rawText === 'string') {
            rawText = responseAny.inference.result.rawText;
        } else if (responseAny.inference.result.rawText.content) {
            rawText = responseAny.inference.result.rawText.content;
        } else if (responseAny.inference.result.rawText.pages) {
            rawText = responseAny.inference.result.rawText.pages.map((page: any) => page.content || page).join('\n\n');
        }
    }
    // Approach 4: Check if there's a pages array with text content
    else if (responseAny.inference && responseAny.inference.pages) {
        console.log('Found pages in inference');
        const pageContents = responseAny.inference.pages.map((page: any, index: number) => {
            if (page.rawText) return page.rawText;
            if (page.content) return page.content;
            if (page.text) return page.text;
            return `Page ${index + 1}: No text content found`;
        });
        rawText = pageContents.join('\n\n');
    }
    
    console.log('Extracted raw text length:', rawText.length);

    // Get structured output
    const structuredResult = response.inference.toString();

    const endTime = Date.now();
    const processingTime = endTime - startTime;

    // Return both raw text and structured result separately
    return {
        rawText: rawText,
        structuredResult: structuredResult,
        fileName: path.basename(inputPath),
        processingTime: processingTime
    };
}

async function main() {
    const filesDir = './files';
    const markdownDir = './markdown_mindee';
    const structuredDir = './mindee_results';

    // Ensure output directories exist
    if (!fs.existsSync(markdownDir)) {
        fs.mkdirSync(markdownDir);
    }
    if (!fs.existsSync(structuredDir)) {
        fs.mkdirSync(structuredDir);
    }

    // Read all files from files directory
    const files = fs.readdirSync(filesDir);

    for (const file of files) {
        const inputPath = path.join(filesDir, file);
        const ext = path.extname(file);
        const baseName = path.basename(file, ext);
        const markdownPath = path.join(markdownDir, `${baseName}_ocr.md`);
        const structuredPath = path.join(structuredDir, `${baseName}_structured.txt`);

        try {
            console.log(`Processing ${file}...`);
            const result = await processFile(inputPath);

            // Save raw text only as markdown with processing time
            const processingTimeSeconds = (result.processingTime / 1000).toFixed(2);
            const rawTextMarkdown = `# Raw Text OCR Results for ${result.fileName}\n\n**Processing Time:** ${processingTimeSeconds} seconds\n**Processed on:** ${new Date().toISOString()}\n\n---\n\n${result.rawText}`;
            fs.writeFileSync(markdownPath, rawTextMarkdown, 'utf8');
            console.log(`Saved raw text to ${markdownPath}`);

            // Save structured output separately
            fs.writeFileSync(structuredPath, result.structuredResult, 'utf8');
            console.log(`Saved structured output to ${structuredPath}`);
        } catch (error) {
            console.error(`Error processing ${file}:`, error);
        }
    }
}

main().catch(console.error);