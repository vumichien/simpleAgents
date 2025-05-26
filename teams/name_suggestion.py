from pathlib import Path
import pandas as pd
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.ollama import Ollama
from dotenv import dotenv_values, load_dotenv
from typing import List
import os
from pydantic import BaseModel, Field
from tqdm import tqdm
import json
from datetime import datetime

load_dotenv()


# Define Pydantic models for structured data
class PredictionResult(BaseModel):
    """Model for single prediction result containing findings and names"""

    findings: str = Field(..., description="Analysis findings from the example data")
    基準名称: str = Field(..., description="Standardized base name")
    標準名称: str = Field(..., description="Standard name")


class PredictionItem(BaseModel):
    """Model for individual prediction item"""

    基準名称: str = Field(..., description="Standardized base name")
    標準名称: str = Field(..., description="Standard name")


class BatchPredictionResult(BaseModel):
    """Model for batch prediction result containing findings and predictions"""

    findings: str = Field(..., description="Analysis findings from the example data")
    predictions: List[PredictionItem] = Field(
        ...,
        description="List of predictions for batch, each containing 基準名称 and 標準名称",
    )


class NameSuggestionAgent:
    """Agent for name suggestion"""

    def __init__(self):
        self.agent = Agent(
            name="NameSuggestionAgent",
            role="Suggests standardized names based on training examples",
            # model=OpenAIChat(id="o4-mini"),
            model=Claude(id="claude-sonnet-4-20250514", structured_outputs=True, max_tokens=10000),
            # model=Ollama(id="gemma3:12b"),
            description="You analyze Japanese construction terminology to suggest standardized names",
            instructions=[
                "You are an expert in Japanese construction terminology standardization.",
                "Your task is to suggest 基準名称 and 標準名称 based on training examples.",
                "Return results in the exact format requested.",
            ],
            response_model=PredictionResult,
            debug_mode=True,
        )
        self.batch_agent = Agent(
            name="BatchNameSuggestionAgent",
            role="Suggests standardized names for multiple items at once",
            # model=OpenAIChat(id="o4-mini", structured_outputs=True),
            model=Claude(id="claude-sonnet-4-20250514", structured_outputs=True, max_tokens=10000),
            # model=Ollama(id="gemma3:12b", structured_outputs=True),
            description="You analyze Japanese construction terminology to suggest standardized names for batches",
            instructions=[
                "You are an expert in Japanese construction terminology standardization.",
                "Your task is to suggest 基準名称 and 標準名称 for multiple items based on training examples.",
                "Return results for ALL items in the exact format requested.",
            ],
            response_model=BatchPredictionResult,
            debug_mode=True,
        )
        self.data_folder = Path("data/data_check_0524")

    def read_examples_from_train(self, train_file_path: Path) -> str:
        """Read training examples from train.csv file"""
        try:
            df = pd.read_csv(train_file_path)
            examples_text = ""

            # If more than 100 rows, take first 25 and last 75
            if len(df) > 100:
                first_25 = df.head(25)
                last_75 = df.tail(75)
                sample_df = pd.concat([first_25, last_75], ignore_index=True)
            else:
                sample_df = df

            for _, row in sample_df.iterrows():
                examples_text += f"名称: {row['名称']}, 摘要: {row.get('摘要', '')}, 備考: {row.get('備考', '')} → 基準名称: {row['基準名称']}, 標準名称: {row['標準名称']}\n"

            return examples_text
        except Exception as e:
            print(f"Error reading train file {train_file_path}: {e}")
            return ""

    def predict_single_row(self, row: pd.Series, examples: str) -> PredictionResult:
        """Predict 基準名称 and 標準名称 for a single row"""
        prompt = f"""Column 基準名称 was created based on judging which phrase is important in column 名称, 摘要 and 備考. Column 標準名称 was created based on 基準名称

There is one fundamental principle to follow: retain the important keywords. While it's not necessary to adhere to past conventions, you must assess which words are significant to keep in order to differentiate them from other groups.

This is example data:
{examples}

First, analyze the example data and provide your findings about the patterns you observe in how 基準名称 and 標準名称 are created from 名称, 摘要, and 備考.

Then, I will provide you with new data that includes only the 名称, 摘要 and 備考 column for you to suggest the corresponding 基準名称, 標準名称 after you read all the current data.

New data to predict:
名称: {row['名称']}
摘要: {row.get('摘要', '')}
備考: {row.get('備考', '')}

Please provide:
1. Your findings about the patterns in the example data
2. The suggested 基準名称 and 標準名称 based on the patterns you learned from the examples."""

        try:
            response = self.agent.run(prompt)
            return response.content
        except Exception as e:
            print(f"Error predicting for row: {e}")
            # Return default values if prediction fails
            return PredictionResult(
                findings="Error occurred during prediction",
                基準名称=row["名称"],
                標準名称=row["名称"],
            )

    def predict_batch(
        self, chunk_df: pd.DataFrame, examples: str
    ) -> tuple[str, List[PredictionItem]]:
        """Predict 基準名称 and 標準名称 for a batch of rows"""
        # Prepare batch data
        batch_data = []
        for _, row in chunk_df.iterrows():
            batch_data.append(
                f"名称: {row['名称']}, 摘要: {row.get('摘要', '')}, 備考: {row.get('備考', '')}"
            )

        batch_text = "\n".join([f"{i+1}. {data}" for i, data in enumerate(batch_data)])

        prompt = f"""Column 基準名称 was created based on judging which phrase is important in column 名称, 摘要 and 備考. Column 標準名称 was created based on 基準名称

There is one fundamental principle to follow: retain the important keywords. While it's not necessary to adhere to past conventions, you must assess which words are significant to keep in order to differentiate them from other groups.

This is example data:
{examples}

First, analyze the example data and provide your findings about the patterns you observe in how 基準名称 and 標準名称 are created from 名称, 摘要, and 備考.

Then, I will provide you with new data that includes only the 名称, 摘要 and 備考 column for you to suggest the corresponding 基準名称, 標準名称 after you read all the current data.

New data to predict (multiple items):
{batch_text}

Please provide:
1. Your findings about the patterns in the example data
2. The suggested 基準名称 and 標準名称 for ALL {len(batch_data)} items based on the patterns you learned from the examples."""

        response = self.batch_agent.run(prompt)
        batch_result = response.content

        # Ensure we have the right number of predictions
        if len(batch_result.predictions) != len(chunk_df):
            raise ValueError(
                f"Expected {len(chunk_df)} predictions, got {len(batch_result.predictions)}"
            )

        return batch_result.findings, batch_result.predictions

    def process_test_data_in_chunks(
        self, test_df: pd.DataFrame, examples: str, chunk_size: int = 50
    ) -> tuple[List[dict], List[str]]:
        """Process test data in chunks if it's large"""
        results = []
        all_findings = []

        # Filter out comment rows first
        non_comment_mask = ~(
            test_df["名称"].isna()
            | test_df["名称"].astype(str).str.strip().str.startswith("（")
        )
        comment_rows = test_df[~non_comment_mask]
        data_rows = test_df[non_comment_mask]

        # Add comment rows to results as-is
        for _, row in comment_rows.iterrows():
            results.append(
                {
                    "名称": row["名称"],
                    "摘要": row.get("摘要", ""),
                    "備考": row.get("備考", ""),
                    "基準名称": row.get("基準名称", ""),
                    "標準名称": row.get("標準名称", ""),
                    "_original_index": row.name,
                }
            )

        # Process data rows in chunks
        if len(data_rows) <= chunk_size:
            # Small dataset, process as single batch
            findings, predictions = self.predict_batch(data_rows, examples)
            all_findings.append(findings)
            for (_, row), prediction in zip(data_rows.iterrows(), predictions):
                results.append(
                    {
                        "名称": row["名称"],
                        "摘要": row.get("摘要", ""),
                        "備考": row.get("備考", ""),
                        "基準名称": prediction.基準名称,
                        "標準名称": prediction.標準名称,
                        "_original_index": row.name,
                    }
                )
        else:
            # Large dataset, process in chunks
            print(
                f"  Processing {len(data_rows)} data rows in chunks of {chunk_size}..."
            )
            for i in range(0, len(data_rows), chunk_size):
                chunk = data_rows.iloc[i : i + chunk_size]
                print(
                    f"    Processing chunk {i//chunk_size + 1}/{(len(data_rows)-1)//chunk_size + 1}"
                )

                findings, predictions = self.predict_batch(chunk, examples)
                all_findings.append(findings)
                for (_, row), prediction in zip(chunk.iterrows(), predictions):
                    results.append(
                        {
                            "名称": row["名称"],
                            "摘要": row.get("摘要", ""),
                            "備考": row.get("備考", ""),
                            "基準名称": prediction.基準名称,
                            "標準名称": prediction.標準名称,
                            "_original_index": row.name,
                        }
                    )

        # Sort results by original index to maintain order
        results.sort(key=lambda x: x["_original_index"])

        # Remove the temporary index field
        for result in results:
            del result["_original_index"]

        return results, all_findings

    def process_subfolder(self, subfolder_path: Path):
        """Process a single subfolder"""
        print(f"\nProcessing subfolder: {subfolder_path.name}")

        train_file = subfolder_path / "train.csv"
        test_file = subfolder_path / "test.csv"
        result_file = subfolder_path / "result.csv"

        # Check if required files exist
        if not train_file.exists():
            print(f"  Warning: train.csv not found in {subfolder_path.name}")
            return

        if not test_file.exists():
            print(f"  Warning: test.csv not found in {subfolder_path.name}")
            return

        # Read examples from train.csv
        print("  Reading examples from train.csv...")
        examples = self.read_examples_from_train(train_file)
        print(f"  Found {len(examples.splitlines())} examples in train.csv")

        if not examples:
            print(f"  Error: Could not read examples from {train_file}")
            return

        # Read test data
        print("  Reading test data...")
        try:
            test_df = pd.read_csv(test_file)
        except Exception as e:
            print(f"  Error reading test file: {e}")
            return

        # Process test data (with chunking if needed)
        print(f"  Processing {len(test_df)} rows...")
        try:
            results, all_findings = self.process_test_data_in_chunks(test_df, examples)
        except Exception as e:
            print(f"  Error processing test data: {e}")
            print(f"  Skipping subfolder {subfolder_path.name}")
            raise e

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(result_file, index=False, encoding="utf-8-sig")
        print(f"  Results saved to {result_file}")

        # Save model output as JSON
        model_output = {
            "subfolder": subfolder_path.name,
            "timestamp": datetime.now().isoformat(),
            "examples_used": examples,
            "findings": all_findings,
            "total_predictions": len(results),
            "predictions": [
                {
                    "名称": row["名称"],
                    "摘要": row["摘要"],
                    "備考": row["備考"],
                    "predicted_基準名称": row["基準名称"],
                    "predicted_標準名称": row["標準名称"],
                }
                for row in results
                if row["基準名称"]
                and row["標準名称"]  # Only include actual predictions, not comments
            ],
            "statistics": {
                "total_rows": len(test_df),
                "comment_rows": len(test_df)
                - len([r for r in results if r["基準名称"] and r["標準名称"]]),
                "prediction_rows": len(
                    [r for r in results if r["基準名称"] and r["標準名称"]]
                ),
                "chunks_processed": len(all_findings),
            },
        }
        json_file = (
            subfolder_path
            / f"model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(model_output, f, ensure_ascii=False, indent=2)
        print(f"  Model output saved to {json_file}")

    def process_all_subfolders(self):
        """Process all subfolders in the data directory"""
        if not self.data_folder.exists():
            print(f"Data folder not found: {self.data_folder}")
            return

        # Get all subdirectories
        subfolders = [f for f in self.data_folder.iterdir() if f.is_dir()]

        if not subfolders:
            print("No subfolders found in data directory")
            return

        print(f"Found {len(subfolders)} subfolders to process")

        # Process each subfolder
        for subfolder in subfolders:
            # if subfolder.name in ["コンクリート", "左官", "塗装"]:
            if subfolder.name not in [
                "コンクリート",
                "その他工作物",
                "ユニット及びその他",
                "免震",
                "共通仮設費",
                "内外装",
                "土工",
                "地業",
                "型枠",
                "塗装",
                "屋外排水",
                "屋根及びとい",
                "左官",
                "建具",
                "既製コンクリート",
                "木工",
            ]:
                try:
                    self.process_subfolder(subfolder)
                except Exception as e:
                    print(f"Error processing subfolder {subfolder.name}: {e}")
                    continue

        print("\nAll subfolders processed!")


if __name__ == "__main__":
    agent = NameSuggestionAgent()
    agent.process_all_subfolders()
