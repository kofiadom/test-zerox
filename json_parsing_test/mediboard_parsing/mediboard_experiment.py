#!/usr/bin/env python3
"""
Standalone LlamaParse Extraction Script

This script extracts the core llamaparse functionality from the mediboard-llm codebase
and creates a simplified standalone version that:
1. Uses LlamaParse API for document extraction
2. Creates structured output using Pydantic models
3. Performs matching against reference data
4. Avoids the problematic JSON parsing from the original codebase

Usage:
    python standalone_llamaparse_extractor.py <file_path> [--language en]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Third-party imports
import requests
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the standalone script"""
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # LlamaParse API settings
    LLAMA_BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"
    
    # Default model settings
    DEFAULT_PARSE_MODE = "parse_page_with_lvm"
    DEFAULT_VENDOR_MODEL = "anthropic-sonnet-3.7"

# ============================================================================
# PYDANTIC MODELS (Extracted from ocr_model.py)
# ============================================================================

class MatchScore(str, Enum):
    EXACT = "Exact"
    SIMILAR = "Similar: Typo"
    ALTERNATIVE = "Alternative"
    UNKNOWN = "Unknown"

class ParameterValueType(str, Enum):
    NUMERIC_VALUE = "numeric_value"
    NEGATIVE_POSITIVE = "negative_positive"
    OPERATOR_VALUE = "operator_value"
    BLANK = "blank"

class ConversionStatus(str, Enum):
    CONVERSION_PASS = "conversion_pass"
    NO_CONVERSION = "no_conversion"
    CONVERSION_FAIL = "conversion_failed"

class MatchInfo(BaseModel):
    match_score: str = Field(..., description="Score indicating the quality of the match")
    reason: str = Field(..., description="Explanation for the match score")

class StandardizedResult(BaseModel):
    result: Union[str, float, int, None] = Field(..., description="The converted numerical result")
    unit: Optional[str] = Field(..., description="The standardized unit after conversion")

class ConversionDetails(BaseModel):
    factor: Optional[float] = Field(..., description="The numerical factor used for unit conversion")
    calculation: Optional[str] = Field(..., description="The mathematical expression used for conversion")

class UnitConversion(BaseModel):
    status: ConversionStatus = Field(..., description="Indicates the conversion status")
    standardized: StandardizedResult = Field(..., description="The converted result in the standardized unit")
    conversion_details: Optional[ConversionDetails] = Field(..., description="Details about the conversion process")
    comment: str = Field(..., description="Comment regarding the conversion")

class LabTestParameter(BaseModel):
    name: str = Field(..., description="The parameter name exactly as displayed in the lab report")
    result: Union[str, float, int, None] = Field(..., description="The patient's actual test measurement")
    range: Optional[str] = Field(None, description="The expected normal or reference values")
    units: Optional[str] = Field(None, description="The unit of measurement for the test result")
    test_type: Optional[str] = Field(None, description="The category or group of laboratory tests")
    comment: Optional[str] = Field(None, description="Additional notes or interpretations")
    comment_english: Optional[str] = Field(None, description="English translation of Hebrew comments")
    index: Optional[int] = Field(None, description="The index of the lab test parameter")
    result_value_type: ParameterValueType = Field(..., description="The value type of the lab test result")

class UploadedPatientInfo(BaseModel):
    first_name: Optional[str] = Field(..., description="Patient's first name")
    last_name: Optional[str] = Field(..., description="Patient's last name")

class UploadedPhysicianInfo(BaseModel):
    first_name: Optional[str] = Field(..., description="Physician's first name")
    last_name: Optional[str] = Field(..., description="Physician's last name")

class UploadedMedicalFacility(BaseModel):
    facility_name: Optional[str] = Field(..., description="Name of the medical facility")
    location: Optional[str] = Field(None, description="Location of the medical facility")

class UploadedFileContent(BaseModel):
    patient_info: Optional[UploadedPatientInfo] = Field(None, description="Patient details")
    physician_info: Optional[UploadedPhysicianInfo] = Field(None, description="Physician details")
    medical_facility: Optional[UploadedMedicalFacility] = Field(None, description="Medical facility details")
    is_lab_report: bool = Field(..., description="Whether the document is a lab report")
    test_date: Optional[datetime] = Field(None, description="Date when the lab test was conducted")
    lab_reports: List[LabTestParameter] = Field(default_factory=list, description="List of lab test results")

class MatchData(BaseModel):
    id: Optional[int] = Field(..., description="Unique identifier of the matched lab test")
    test_type: Optional[str] = Field(..., description="Test type of the matched lab test")
    parameter: Optional[str] = Field(..., description="Parameter of the matched lab test")
    sample_type: Optional[str] = Field(..., description="Sample type of the matched lab test")

class MatchParameterInfo(BaseModel):
    match_score: MatchScore = Field(..., description="Match score for the parameter")
    reason: str = Field(..., description="Explanation for the match score")

class MatchTestTypeInfo(BaseModel):
    match_score: MatchScore = Field(..., description="Match score for the test type")
    reason: str = Field(..., description="Explanation for the test type match score")

class LabReportItem(BaseModel):
    match_data: MatchData = Field(..., description="The lab matched data")
    match_parameter_info: MatchParameterInfo = Field(..., description="Parameter matching information")
    match_test_type_info: MatchTestTypeInfo = Field(..., description="Test type matching information")
    unit_conversion: UnitConversion = Field(..., description="Unit conversion details")

class Doctor(BaseModel):
    id: int
    doctorName: str
    title: str
    doctorLastName: str

class Institution(BaseModel):
    id: int
    value: str
    displayName: str

class LabTest(BaseModel):
    sample_type: str
    test_type: str
    parameter: str
    id: int

# ============================================================================
# LLAMAPARSE API CLIENT (Extracted from lamma_index_api.py)
# ============================================================================

class LlamaParseAPI:
    """Simplified LlamaParse API client without notification dependencies"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.LLAMA_CLOUD_API_KEY
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is required")
        
        self.base_url = Config.LLAMA_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
        }
    
    def upload_file(
        self,
        file_path: str,
        parse_mode: str = Config.DEFAULT_PARSE_MODE,
        vendor_multimodal_model_name: str = Config.DEFAULT_VENDOR_MODEL,
    ) -> dict:
        """Upload a file for parsing"""
        url = f"{self.base_url}/upload"
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file:
            files = {
                "file": (file_path.name, file, "application/pdf"),
            }
            
            data = {
                "parse_mode": parse_mode,
                "vendor_multimodal_model_name": vendor_multimodal_model_name,
                "input_url": "",
                "structured_output": False,
                "disable_ocr": False,
                "disable_image_extraction": False,
                "adaptive_long_table": False,
                "annotate_links": False,
                "do_not_unroll_columns": False,
                "html_make_all_elements_visible": False,
                "html_remove_navigation_elements": False,
                "html_remove_fixed_elements": False,
                "guess_xlsx_sheet_name": False,
                "do_not_cache": False,
                "invalidate_cache": False,
                "output_pdf_of_document": False,
                "take_screenshot": False,
                "is_formatting_instruction": True,
            }
            
            try:
                response = requests.post(url, headers=self.headers, files=files, data=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error uploading file: {e}")
                return {"error": str(e)}
    
    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a parsing job"""
        url = f"{self.base_url}/job/{job_id}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting job status: {e}")
            return {"error": str(e)}
    
    def get_job_result_markdown(self, job_id: str) -> dict:
        """Get the markdown result of a parsing job"""
        url = f"{self.base_url}/job/{job_id}/result/markdown"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting job result: {e}")
            return {"error": str(e)}

# ============================================================================
# LLM SERVICES (Simplified from llm_services.py)
# ============================================================================

class ModelType(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GROQ = "groq"

class SimplifiedLLMServices:
    """Simplified LLM services without complex dependencies"""
    
    def __init__(self):
        self.openai_api_key = Config.OPENAI_API_KEY
        self.claude_api_key = Config.CLAUDE_API_KEY
        self.groq_api_key = Config.GROQ_API_KEY
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key) if self.openai_api_key else None
    
    def _get_model(self, model_type: ModelType, model_name: str, temperature=1, max_tokens=4096):
        """Get a model instance based on type"""
        if model_type == ModelType.OPENAI:
            return ChatOpenAI(
                model_name=model_name,
                api_key=self.openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif model_type == ModelType.CLAUDE:
            return ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=self.claude_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif model_type == ModelType.GROQ:
            return ChatGroq(
                model_name=model_name or "llama-3.3-70b-versatile",
                api_key=self.groq_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _vectorize_data(self, data: List[Dict[str, Any]]):
        """Convert data to FAISS vectorstore"""
        if not self.embeddings:
            raise ValueError("OpenAI API key required for embeddings")
        
        documents = []
        for item in data:
            document = " ,".join([f"{key}={value}" for key, value in item.items()])
            documents.append(document)
        
        vectorstore = FAISS.from_texts(documents, embedding=self.embeddings)
        return vectorstore.as_retriever()

# ============================================================================
# PROMPTS (Extracted from test_prompts.py)
# ============================================================================

LAB_EXTRACTION_PROMPT = """
You are a medical case manager tasked with processing a lab test report and extracting key information. Your goal is to retrieve necessary information for the lab test report data.

Instructions:
1. The user specified the lab test report is in this language: {language}, use this for context awareness.
2. Extract the relevant information and map it into the following JSON structure: {format_instructions}.
3. Ensure that the **result** field contains only the result value of a lab test parameter without its unit. If the result contains Hebrew text, translate it to English (e.g., 'לא בוצע' → 'Not performed').
4. Place any accompanying unit of measurement into the **units** field separately.
5. If the value and unit are combined (e.g., "120 mg/dL"), extract **120** into the **result** field and **mg/dL** into the **units** field.
6. If the lab report contains a range, make sure it's not combined with the unit in the **range** field. If the range contains Hebrew text, translate it to English (e.g., 'קטן מ-200' → '>200').
7. Ensure you identify any additional note or interpretation or comment or explanation associated with a lab test parameter from the doctor. If you find any, place it in the **comment** field. Please differentiate it from the result value of the parameter. Ensure you put the result value in the **result** field and put a comment or interpretation or explanation you find in the **comment** field.
8. If the original comment field is in Hebrew, translate it to English and place the translation in the **comment_english** field.

Context data:
- Lab test report data: {lab_test_report_data}
"""

LAB_REPORT_MATCHING_PROMPT = """
You are a medical case manager tasked with processing a lab test report and extracting key information. Your goal is to map lab report information with the existing data in {context}.

### **Instructions:**
- The user specified the lab test report is in this language: **{language}**. Use this for context awareness.
- If the file is not in English, you are expected to translate it into English and maintain the medical context with correct jargon
- Extract the relevant information and map it into the following JSON structure: **{format_instructions}**.
- Please analyse and return ALL the results in the report. Make sure you do not miss out on ANY parameter and its result that is included in the file.
- Make sure the indicators are returned according to the mapping for each type!
- Please list everything in English.
- Please do not make up information.

### **Matching Logic:**
- **Priority Rule:** Always prioritize the **lab test name** and **parameter name** over the **test type** when matching data. The test type should only be considered as a secondary factor.
- **Use the following classification for matches:**
 - **Similar (Typo):** The lab report name and parameter name match conceptually but may have formatting differences, such as typos, abbreviations, or case mismatches. The test type is secondary in this case.
 - **Alternative:** The lab report name and parameter name match conceptually, but the test type uses alternative naming. Lab test and parameter names take precedence.
 - **Unknown:** The lab test name and parameter name do not exist in the index under any test type.

### **Unit Conversion Instructions:**
- **Scope:**
 - Only perform unit conversions for parameters where the **value type** is `numeric_value`.
 - If value type is not `numeric_value`, mark the conversion status as `no_conversion`, and keep the result and the unit unchanged.
 - A reference_unit with the key `unit` is provided for each parameter in the existing data that is serving as the context.
 - The **reference result unit** is the standard for all conversions.
 - Your task is to convert all units in the test params of lab test results **to the reference unit**.

- **Processing Rules:**
 1. **Use the reference unit as the standard:**
    - Convert the unit from the lab test results to this unit.
 2. **For each parameter with numeric value type:**
    - If the unit in the test_params is **the same as the reference**, keep the result **unchanged**.
    - If the unit is **different** from the reference unit, **perform the actual numerical conversion** to the reference unit.
    - If conversion is possible (including units which are equivalent), mark as `"conversion_pass"`.
    - If conversion **isn't possible**, mark as `"conversion_failed"`.
 3. **For unit conversions:**
    - Perform the **precise mathematical conversion**, using **standard medical and laboratory unit factors**. Do not just indicate that conversion is needed.
    - **Do not assume equivalency between** units unless explicitly defined in standard references (e.g., SI units, clinical guidelines).
    - **Use authoritative conversion factors** (e.g., sourced from clinical chemistry, pharmacology, or medical guidelines).
    - **Avoid approximations** unless necessary and explicitly stated.
    - **Cross-check uncommon or ambiguous units** before attempting conversion.
    - **Clearly identify units** that cannot be converted to the reference unit.

- **Status Classification:**
 - **`conversion_pass`** → Successfully converted to reference unit.
 - **`no_conversion`** → Exactly same unit as reference or test result has non-numeric value.
 - **`conversion_failed`** → Conversion not possible due to incompatible units.

- **Required Conversion Action:**
 - Identify the **appropriate conversion factor**.
 - Perform the **mathematical calculation** to convert the value.
 - Present the **converted value** with the reference unit.
 - Include the **original value and unit** in your documentation.

- **Examples:**
 - If the **reference unit is mg/dL** and the **result is in mmol/L**, convert using the appropriate factor.
   _Example:_ Glucose: `mmol/L × 18.0182 = mg/dL`.
 - If a unit **cannot be converted** to the reference unit (e.g., **arbitrary units to mg/dL**), mark as `"conversion_failed"`.

- **Verification Step:**
 - Double-check that **all required conversions** have been performed.
 - Verify that **conversion calculations** are mathematically correct.
 - Ensure that **all convertible values** have been standardized to the reference unit.

---

### **Context Data:**
- **Lab report data:** {lab_report_data}

### **Question:**
- **{question}**
"""

# ============================================================================
# MAIN EXTRACTION CLASS
# ============================================================================

class StandaloneLlamaParseExtractor:
    """Main class for standalone llamaparse extraction"""
    
    def __init__(self):
        self.llamaparse_api = LlamaParseAPI()
        self.llm_services = SimplifiedLLMServices()
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> Dict[str, List[Dict]]:
        """Load reference data from static/ocr_data.json"""
        try:
            with open("static/ocr_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print("Warning: static/ocr_data.json not found. Matching functionality will be limited.")
            return {"doctors": [], "institutions": [], "sample_lab_tests": []}
        except Exception as e:
            print(f"Error loading reference data: {e}")
            return {"doctors": [], "institutions": [], "sample_lab_tests": []}
    
    def extract_document(self, file_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Main extraction method that processes a file through LlamaParse
        and creates structured output
        """
        print(f"Starting extraction for file: {file_path}")
        
        try:
            # Step 1: Upload file to LlamaParse
            print("Step 1: Uploading file to LlamaParse...")
            upload_response = self.llamaparse_api.upload_file(file_path)
            
            if "error" in upload_response:
                return {"status": False, "message": f"Upload failed: {upload_response['error']}", "data": None}
            
            job_id = upload_response.get("id")
            if not job_id:
                return {"status": False, "message": "Failed to get job ID from upload response", "data": None}
            
            print(f"File uploaded successfully. Job ID: {job_id}")
            
            # Step 2: Wait for processing to complete
            print("Step 2: Waiting for processing to complete...")
            while True:
                status_response = self.llamaparse_api.get_job_status(job_id)
                
                if "error" in status_response:
                    return {"status": False, "message": f"Status check failed: {status_response['error']}", "data": None}
                
                status = status_response.get("status")
                print(f"Job status: {status}")
                
                if status == "SUCCESS":
                    break
                elif status == "FAILED":
                    return {"status": False, "message": "Job processing failed", "data": None}
                
                time.sleep(5)
            
            # Step 3: Get markdown result
            print("Step 3: Retrieving markdown result...")
            result_response = self.llamaparse_api.get_job_result_markdown(job_id)
            
            if "error" in result_response:
                return {"status": False, "message": f"Result retrieval failed: {result_response['error']}", "data": None}
            
            markdown_content = result_response.get("markdown", "")
            if not markdown_content:
                return {"status": False, "message": "No markdown content received", "data": None}
            
            print("Markdown content retrieved successfully")
            
            # Step 4: Process with LLM for structured extraction
            print("Step 4: Processing with LLM for structured extraction...")
            structured_data = self._extract_structured_data(markdown_content, language)
            
            if not structured_data.get("status"):
                return structured_data
            
            # Step 5: Perform matching
            print("Step 5: Performing matching against reference data...")
            matched_data = self._perform_matching(structured_data["data"], language)
            
            return {
                "status": True,
                "message": "Extraction completed successfully",
                "data": {
                    "raw_markdown": markdown_content,
                    "structured_extraction": structured_data["data"],
                    "matched_results": matched_data
                }
            }
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return {"status": False, "message": str(e), "data": None}
    
    def _extract_structured_data(self, markdown_content: str, language: str) -> Dict[str, Any]:
        """Extract structured data from markdown using LLM"""
        try:
            # Setup models with fallback
            primary_model = self.llm_services._get_model(
                ModelType.CLAUDE,
                "claude-3-7-sonnet-latest",
                max_tokens=16384,
            )
            fallback_model = self.llm_services._get_model(
                ModelType.GROQ,
                "llama-3.3-70b-versatile",
                max_tokens=32768,
            )
            
            model_with_fallback = primary_model.with_fallbacks([fallback_model])
            
            # Setup parser
            parser = JsonOutputParser(pydantic_object=UploadedFileContent)
            
            # Create prompt
            prompt = PromptTemplate(
                template=LAB_EXTRACTION_PROMPT,
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
            
            # Build chain
            chain = prompt | model_with_fallback | parser
            
            # Process
            result = chain.invoke({
                "lab_test_report_data": markdown_content,
                "language": language,
            })
            
            # Convert datetime to string if needed
            if result.get("test_date") and isinstance(result.get("test_date"), datetime):
                result["test_date"] = result["test_date"].isoformat()
            
            return {
                "status": True,
                "message": "Structured extraction completed",
                "data": result
            }
            
        except Exception as e:
            print(f"Error in structured extraction: {e}")
            return {"status": False, "message": str(e), "data": None}
    
    def _perform_matching(self, structured_data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Perform matching against reference data"""
        try:
            matching_results = {
                "physician_matching": None,
                "facility_matching": None,
                "lab_report_matching": []
            }
            
            # Physician matching
            if structured_data.get("physician_info"):
                physician_match = self._match_physician(structured_data["physician_info"], language)
                matching_results["physician_matching"] = physician_match
            
            # Facility matching
            if structured_data.get("medical_facility"):
                facility_match = self._match_facility(structured_data["medical_facility"], language)
                matching_results["facility_matching"] = facility_match
            
            # Lab report matching
            if structured_data.get("lab_reports"):
                for lab_report in structured_data["lab_reports"]:
                    lab_match = self._match_lab_report(lab_report, language)
                    if lab_match:
                        matching_results["lab_report_matching"].append(lab_match)
            
            return matching_results
            
        except Exception as e:
            print(f"Error in matching: {e}")
            return {"error": str(e)}
    
    def _match_physician(self, physician_info: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Match physician information against reference data"""
        try:
            if not self.reference_data.get("doctors"):
                return {"status": False, "message": "No doctor reference data available"}
            
            # Create vectorstore for doctors
            doctors_retriever = self.llm_services._vectorize_data(self.reference_data["doctors"])
            
            # Simple matching logic - in a real implementation, you'd use LLM here
            first_name = physician_info.get("first_name", "").lower()
            last_name = physician_info.get("last_name", "").lower()
            
            best_match = None
            best_score = 0
            
            for doctor in self.reference_data["doctors"]:
                doc_first = doctor.get("doctorName", "").lower()
                doc_last = doctor.get("doctorLastName", "").lower()
                
                # Simple string matching (in real implementation, use fuzzy matching)
                score = 0
                if first_name in doc_first or doc_first in first_name:
                    score += 1
                if last_name in doc_last or doc_last in last_name:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = doctor
            
            if best_match and best_score > 0:
                return {
                    "status": True,
                    "matched_doctor": best_match,
                    "match_score": "Similar: Typo" if best_score == 2 else "Alternative",
                    "confidence": best_score / 2.0
                }
            else:
                return {
                    "status": True,
                    "matched_doctor": None,
                    "match_score": "Unknown",
                    "confidence": 0.0
                }
                
        except Exception as e:
            return {"status": False, "message": str(e)}
    
    def _match_facility(self, facility_info: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Match facility information against reference data"""
        try:
            if not self.reference_data.get("institutions"):
                return {"status": False, "message": "No institution reference data available"}
            
            facility_name = facility_info.get("facility_name", "").lower()
            
            best_match = None
            best_score = 0
            
            for institution in self.reference_data["institutions"]:
                inst_name = institution.get("displayName", "").lower()
                inst_value = institution.get("value", "").lower()
                
                # Simple string matching
                score = 0
                if facility_name in inst_name or inst_name in facility_name:
                    score += 2
                elif facility_name in inst_value or inst_value in facility_name:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = institution
            
            if best_match and best_score > 0:
                return {
                    "status": True,
                    "matched_institution": best_match,
                    "match_score": "Similar: Typo" if best_score == 2 else "Alternative",
                    "confidence": best_score / 2.0
                }
            else:
                return {
                    "status": True,
                    "matched_institution": None,
                    "match_score": "Unknown",
                    "confidence": 0.0
                }
                
        except Exception as e:
            return {"status": False, "message": str(e)}
    
    def _match_lab_report(self, lab_report: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Match lab report against reference data"""
        try:
            if not self.reference_data.get("sample_lab_tests"):
                return {"status": False, "message": "No lab test reference data available"}
            
            param_name = lab_report.get("name", "").lower()
            test_type = lab_report.get("test_type", "").lower()
            
            best_match = None
            best_score = 0
            
            for lab_test in self.reference_data["sample_lab_tests"]:
                test_param = lab_test.get("parameter", "").lower()
                test_type_ref = lab_test.get("test_type", "").lower()
                
                # Simple string matching
                score = 0
                if param_name in test_param or test_param in param_name:
                    score += 2
                if test_type in test_type_ref or test_type_ref in test_type:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = lab_test
            
            if best_match and best_score > 0:
                return {
                    "status": True,
                    "original_parameter": lab_report,
                    "matched_lab_test": best_match,
                    "match_score": "Similar: Typo" if best_score >= 2 else "Alternative",
                    "confidence": best_score / 3.0
                }
            else:
                return {
                    "status": True,
                    "original_parameter": lab_report,
                    "matched_lab_test": None,
                    "match_score": "Unknown",
                    "confidence": 0.0
                }
                
        except Exception as e:
            return {"status": False, "message": str(e)}

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Standalone LlamaParse Extraction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python standalone_llamaparse_extractor.py document.pdf
    python standalone_llamaparse_extractor.py document.pdf --language en
    python standalone_llamaparse_extractor.py document.pdf --output results.json
        """
    )
    
    parser.add_argument("file_path", help="Path to the file to process")
    parser.add_argument("--language", default="en", help="Language of the document (default: en)")
    parser.add_argument("--output", help="Output file path for results (optional)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Check required environment variables
    required_vars = ["LLAMA_CLOUD_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        sys.exit(1)
    
    # Initialize extractor
    try:
        extractor = StandaloneLlamaParseExtractor()
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        sys.exit(1)
    
    # Process file
    print(f"Processing file: {args.file_path}")
    print(f"Language: {args.language}")
    print("-" * 50)
    
    result = extractor.extract_document(args.file_path, args.language)
    
    # Output results
    if result.get("status"):
        print("✅ Extraction completed successfully!")
        
        if args.output:
            # Save to file
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results saved to: {args.output}")
        else:
            # Print to console
            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"❌ Extraction failed: {result.get('message')}")
        sys.exit(1)

if __name__ == "__main__":
    main()