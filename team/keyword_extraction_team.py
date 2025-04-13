from pathlib import Path
import pandas as pd
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.ollama import Ollama
from dotenv import dotenv_values
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm

config = dotenv_values(".env")

# Define more detailed example patterns for training
EXAMPLES = """
Examples of extraction patterns:

Basic extraction (all words not in 基準名称 are unimportant, separated by brackets):
- 名称: #階ﾃﾗｽ床人工木ﾃﾞｯｷ部屋外ﾍﾞﾝﾁ → 基準名称: ﾍﾞﾝﾁ → Unimportant: [#階][ﾃﾗｽ][床][人工木][ﾃﾞｯｷ部][屋外]

Multiple unconnected unimportant parts (separated by brackets):
- 名称: #m超軽量鉄骨壁下地補強材 → 基準名称: 軽量鉄骨壁下地 → Unimportant: [#m超][補強材]

Number replacement (# represents any number):
- 名称: AW-#欄間ｶﾞﾗﾘ+突出し換気窓付引違い → 基準名称: AW-#窓 → Unimportant: [欄間][ｶﾞﾗﾘ][突出し換気][窓付][引違い]
- 名称: AW-#引違い窓 → 基準名称: AW-#窓 → Unimportant: [引違い]
- 名称: LD-#a小窓付片開扉 → 基準名称: LD-#扉 → Unimportant: [#a][小窓付][片開]
- 名称: LD-#ｽﾘｯﾄ付片開扉 → 基準名称: LD-#扉 → Unimportant: [#ｽﾘｯﾄ付][片開]

Special cases with punctuation (separate at punctuation marks):
- 名称: E-#EXP_J~EV枠取合ﾊﾟﾈﾙ → 基準名称: 取合ﾊﾟﾈﾙ → Unimportant: [E][#EXP_J][EV枠]
- 名称: ｸﾘｱﾗｯｶｰ塗り(CL) → 基準名称: CL塗り → Unimportant: [ｸﾘｱﾗｯｶ]
- 名称: 地下#階駐車場 → 基準名称: 駐車場金属 → Unimportant: [地下#階]

Important terms in different positions:
- 名称: HWCｽﾃﾝﾚｽ面台 → 基準名称: WC面台 → Unimportant: [H][ｽﾃﾝﾚｽ]
- 名称: 階段H袖ﾊﾟﾈﾙ → 基準名称: ﾊﾟﾈﾙ → Unimportant: [階段][H袖]
- 名称: 風除室天井ｱﾙﾐｶｯﾄﾊﾟﾈﾙ → 基準名称: 天井ｱﾙﾐｶｯﾄﾊﾟﾈﾙ → Unimportant: [風除室]
"""


# Define Pydantic models for structured data
class KeywordPair(BaseModel):
    processed: str = Field(..., description="Processed text (処理名称)")
    standard: str = Field(..., description="Standardized text (基準名称)")


class ExtractedKeyword(BaseModel):
    unimportant_parts: str = Field(
        ...,
        description="The unimportant parts that were removed during standardization, with each part in brackets like [part1][part2]",
    )


class KeywordResult(BaseModel):
    名称: str = Field(..., description="Original text")
    処理名称: str = Field(..., description="Processed text")
    基準名称: str = Field(..., description="Standardized text")
    不要部分: str = Field(..., description="Unimportant parts extracted, in brackets")


class KeywordExtractor:
    def __init__(self, input_file: str, output_file: str):
        self.extractor = Agent(
            name="Extractor",
            role="Extracts unimportant parts from text pairs",
            model=Ollama(id="gemma3:12b"),
            description="You analyze pairs of original and standardized Japanese text to identify unimportant parts",
            instructions=[
                "Given a pair of original text (名称) and standardized text (基準名称):",
                "1. Identify ALL parts that were removed or changed in the standardization process",
                "2. Put each unimportant part in brackets like [part1][part2][part3]",
                "3. Break down unimportant parts into meaningful chunks/words, not individual characters",
                "4. Separate words at punctuation marks like -, +, ~, _, ., etc.",
                "5. When # appears in 基準名称, it represents any number in the original text",
                "6. For cases with prefixes like AW-, SD-, etc., handle them as separate parts",
                "7. Pay attention to where the important terms appear - the unimportant parts can be before, within, or after the important terms",
                "8. Return ONLY the unimportant parts in brackets, nothing else - no explanations",
                "9. Use the provided examples as a guide for extraction",
                EXAMPLES,
            ],
            response_model=ExtractedKeyword,
        )
        self.input_file = input_file
        self.output_file = output_file

    def process_file(self):
        # Read input CSV
        df = pd.read_csv(self.input_file)
        results = []

        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            original = row["名称"] if "名称" in row else row["keyword"]
            standard = row["基準名称"]
            processed = row["処理名称"]

            # Create structured input
            keyword_input = KeywordPair(processed=processed, standard=standard)

            # Get extractor response with structured output
            prompt = f"""Extract unimportant parts from:
名称: {processed}
基準名称: {standard}

Identify ALL unimportant parts. Put each part in brackets like [part1][part2]. 
When # appears in 基準名称, it represents any number in the original.
Separate words at punctuation marks like -, +, ~, _, ., etc.

Important notes:
1. The important terms (from 基準名称) can appear at the beginning, middle, or end of the original text
2. You must identify all parts that are NOT in the 基準名称
3. Break down unimportant parts into logical units

Examples:
'E-#EXP_J~EV枠取合ﾊﾟﾈﾙ' with important part '取合ﾊﾟﾈﾙ' → '[E][#EXP_J][EV枠]'
'HWCｽﾃﾝﾚｽ面台' with important part 'WC面台' → '[H][ｽﾃﾝﾚｽ]'"""
            response = self.extractor.run(prompt)

            # The response.content is already structured as ExtractedKeyword
            extracted = response.content

            # Add to results
            results.append(
                KeywordResult(
                    名称=original,
                    処理名称=processed,
                    基準名称=standard,
                    不要部分=extracted.unimportant_parts,
                )
            )

        # Convert to DataFrame and save
        results_dict = [item.model_dump() for item in results]
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(self.output_file, index=False, encoding="utf-8")
        print(f"Processing completed. Results saved to {self.output_file}")


if __name__ == "__main__":
    input_file = "preprocess_keyword.csv"
    output_file = "extraction_results_ollama.csv"

    extractor = KeywordExtractor(input_file, output_file)
    extractor.process_file()
