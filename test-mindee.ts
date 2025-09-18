const mindee = require("mindee");
// for TS or modules:
// import * as mindee from "mindee";

const apiKey = "md_3ivi793hFgEG7bjpYebaJqy7ACfyXHPZ";
const filePath = "./files/file_1.jpg";
const modelId = "e0c71f16-ffde-448a-8973-dfa878a6ede8";

// Init a new client
const mindeeClient = new mindee.ClientV2({ apiKey: apiKey });

// Set inference parameters
const inferenceParams = {
  modelId: modelId,

  // Options: set to `true` or `false` to override defaults

  // Enhance extraction accuracy with Retrieval-Augmented Generation.
  rag: undefined,
  // Extract the full text content from the document as strings.
  rawText: true,
  // Calculate bounding box polygons for all fields.
  polygon: undefined,
  // Boost the precision and accuracy of all extractions.
  // Calculate confidence scores for all fields.
  confidence: undefined,
};

// Load a file from disk
const inputSource = new mindee.PathInput({ inputPath: filePath });

// Send for processing
const response = mindeeClient.enqueueAndGetInference(
  inputSource,
  inferenceParams
);

// Handle the response Promise
response.then((resp: any) => {
  // print a string summary
  console.log(resp.inference.toString());
  
  // Debug: Log the entire response structure
  console.log("\n=== DEBUG: Response Structure ===");
  console.log("Full response:", JSON.stringify(resp, null, 2));
  
  // Handle raw text extraction
  const rawText = resp.inference.result.rawText;
  
  if (rawText) {
    console.log("\n=== RAW TEXT EXTRACTION ===");
    
    // get the entire document as a single string
    const documentText = rawText.toString();
    console.log("Full document text:");
    console.log(documentText);
    
    // loop over pages
    console.log("\n=== PAGE-BY-PAGE TEXT ===");
    for (const page of rawText.pages) {
      const pageText = page.content;
      console.log(`Page ${page.pageNumber || 'N/A'}:`);
      console.log(pageText);
      console.log("---");
    }
  } else {
    console.log("No raw text available. Make sure rawText parameter is enabled.");
    console.log("Debug - resp.inference.result:", resp.inference.result);
  }
}).catch((error: any) => {
  console.error("Error processing document:", error);
});