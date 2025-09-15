# OCR Multiple Files Script

A comprehensive OCR (Optical Character Recognition) solution that processes multiple files using AWS Bedrock AI models and Zerox library. Extracts text from documents and saves results in both JSON and Markdown formats.

## Features

- 🚀 **Batch Processing**: Process multiple files simultaneously
- 📄 **Multiple Formats**: Supports PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP
- 🤖 **AI-Powered**: Uses Anthropic Claude models via AWS Bedrock
- 📊 **Dual Output**: Generates both JSON and formatted Markdown files
- ⚡ **Performance**: Concurrent processing with configurable concurrency
- 🔄 **Retry Logic**: Automatic retry on failures
- 📈 **Progress Tracking**: Real-time processing status and timing

## Prerequisites

- **Node.js** (v18 or higher)
- **Ghostscript** (for PDF processing)
- **GraphicsMagick** (for image processing)
- **LibreOffice** (for document conversion)
- **AWS Bedrock** access with appropriate permissions

## Installation

### 1. Install System Dependencies

**Ghostscript** (Required for PDF processing):
```bash
# Download from: https://www.ghostscript.com/download.html
# Install and ensure it's in your system PATH
```

**GraphicsMagick** (Required for image processing):
```bash
# Download from: http://www.graphicsmagick.org/download.html
# Install and ensure it's in your system PATH
```

**LibreOffice** (Required for document processing):
```bash
# Download from: https://www.libreoffice.org/download/download/
# Install and ensure soffice command is in your system PATH
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Environment Setup

Create a `.env` file in the project root:

```env
# AWS Bedrock Configuration
BEDROCK_AWS_ACCESS_KEY_ID=your_access_key_here
BEDROCK_AWS_SECRET_ACCESS_KEY=your_secret_key_here
BEDROCK_AWS_SESSION_TOKEN=your_session_token_here
BEDROCK_AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20240620-v1:0

# Optional: S3 Configuration (for test_s3_fetch.ts)
AWS_ACCESS_KEY_ID=your_s3_access_key
AWS_SECRET_ACCESS_KEY=your_s3_secret_key
AWS_REGION=eu-central-1
S3_BUCKET=your_bucket_name
```

> ⚠️ **Security Warning**: Never commit your `.env` file to version control!
> Create a `.env.example` file as a template for others.

## Environment Setup

Create a `.env` file in the project root with your AWS Bedrock credentials:

```env
BEDROCK_AWS_ACCESS_KEY_ID=your_access_key
BEDROCK_AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_AWS_REGION=eu-west-1
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Usage

### Main OCR Script

1. **Prepare your files**:
   ```bash
   # Create files directory
   mkdir files

   # Add your files to process
   # Supported formats: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP
   ```

2. **Run the OCR script**:
   ```bash
   npx ts-node ocr-multiple-files.ts
   ```

### Test Bedrock Connection

Before processing large batches, test your Bedrock connection:
```bash
npx ts-node test-bedrock-inference.ts
```

### S3 File Fetching (Optional)

If you have files in S3, use the fetch script:
```bash
npx ts-node test_s3_fetch.ts
```

## Output

The script generates two types of output files:

### JSON Output (`ocr_output/` directory)
- Detailed JSON results with metadata
- Token usage and processing statistics
- Page-by-page extraction results
- Example: `document.pdf` → `document_ocr.json`

### Markdown Output (`markdown_extraction/` directory)
- Human-readable formatted markdown files
- Processing summary and metadata
- Content organized by page
- Example: `document.pdf` → `document_ocr.md`

### Output Features
- **Processing Time**: Displayed in seconds (e.g., "11.07s")
- **Token Tracking**: Input/output token counts
- **Page Organization**: Content separated by page
- **Error Handling**: Detailed error reporting for failed files

## Supported Models

The script uses AWS Bedrock models. Default is `anthropic.claude-3-5-sonnet-20241022-v2:0`. You can change this in your `.env` file.

Other supported models include:
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`

## Configuration

You can modify the script for:
- **Concurrency**: Change the `concurrency` parameter (default: 5)
- **Retries**: Adjust `maxRetries` (default: 3)
- **Output directory**: Modify `OUTPUT_DIR` variable

## Troubleshooting

- **"Ghostscript not found"**: Ensure Ghostscript is installed and in PATH
- **AWS credentials error**: Check your `.env` file and AWS permissions
- **No files found**: Ensure files are in the `files` directory with supported extensions

## Example Output Structure

```json
{
  "completionTime": 10038,
  "fileName": "document",
  "inputTokens": 25543,
  "outputTokens": 210,
  "pages": [
    {
      "page": 1,
      "content": "# Extracted Text Content...",
      "contentLength": 747
    }
  ],
  "summary": {
    "totalPages": 1,
    "ocr": {
      "failed": 0,
      "successful": 1
    }
  }
}