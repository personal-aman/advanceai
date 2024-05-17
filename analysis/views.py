import re

from typing import List, Dict
from dotenv import load_dotenv

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from analysis.aiService.constants import ADDITIONAL_INFORMATION_FOR_CLASSIFICATION, get_important_note
from analysis.aiService.weaviateDb import fetch_top_k_content, storeData
from analysis.serializers import TranscriptionSerializer
from analysis.models import Transcription, StatementClassification, StatementClassificationTypePrompt, llmModel, StatementLevelPrompt, FinalStatementWithLevel
from analysis.utils import create_overlapping_segments

from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.globals import set_debug
from langchain_community.llms.openai import AzureOpenAI

load_dotenv()

# set_debug(True)
class StatementParser(BaseModel):
    opening_statements: List[str] = Field(description="list of different Opening statements")
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")
    closing_outcome_sentences: List[Dict[str, str]] = Field(
        description="list of dictionaries containing closing statement of REP ('closing_statement') and outcome statements OF HCP ('outcome_statement')")

class StatementParserWithoutOpening(BaseModel):
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")
    closing_outcome_sentences: List[Dict[str, str]] = Field(
        description="list of dictionaries containing closing and outcome statements. Example: {'closing_statement': closing_dailouge (REP), 'outcome_statement': outcome_dailouge(HCP)}")

class StatementParserWithoutClosing(BaseModel):
    opening_statements: List[str] = Field(description="list of different Opening statements")
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")

class StatementParserWithoutOpeningAndClosing(BaseModel):
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")

class StatementLevelModel(BaseModel):
    id: int = Field(description="this field is the ID of the statement")
    level: int = Field(description="this field is the assigned level (1-4)")
    confidence_score: int = Field(description="this field is your confidence score, indicating how certain you are about your evaluation.")
    reason: int = Field(description="reason for you evaluation in max 1 line")

class EvaluationResult(BaseModel):
    statements: List[StatementLevelModel] = Field(..., description="A list of statement level Model")

def get_statements_data(part, total_parts):
    statement_types = StatementClassificationTypePrompt.objects.filter(active=True)
    prompt_template = (" category: {category} , where "
              "the definition of the {category} category: ###{definition}### \n"
              "the examples of the {category} category: ###{examples}### \n"
              )
    prompt = PromptTemplate(
        input_variables=[
            'category',
            'definition',
            'examples'
        ],
        template=prompt_template,
    )
    total_types = []
    final_prompt = ""
    for statement_type in statement_types:
        if (total_parts == 2):
            if(part == 2) and (statement_type.category == "OPENING"):
                continue
            if (part == 1) and (statement_type.category == "CLOSING_OUTCOME"):
                continue
        if(total_parts >= 3 and (((part) * 1.0) / total_parts > .3)) and (statement_type.category == "OPENING"):
            continue
        if(total_parts >= 3 and (((part) * 1.0) / total_parts < .7)) and (statement_type.category == "CLOSING_OUTCOME"):
            continue

        print(statement_type.category)
        total_types.append(statement_type.category)
        final_prompt += "\n-----------for "+statement_type.category+" statements -----------\n"
        prompt_value = prompt.format_prompt(
            category=statement_type.category,
            definition=statement_type.definition,
            examples=statement_type.examples
        )
        final_prompt += "'''" + str(prompt_value) + "'''"

    return total_types, final_prompt

def get_important_note_and_parser(total_type_considered, segment, total_segments):

    important_note = get_important_note(not_included_statements="")
    parser = PydanticOutputParser(pydantic_object=StatementParser)

    if "OPENING" not in total_type_considered:
        parser = PydanticOutputParser(pydantic_object=StatementParserWithoutOpening)
        important_note = get_important_note(not_included_statements="OPENING")

    if "CLOSING_OUTCOME" not in total_type_considered:
        parser = PydanticOutputParser(pydantic_object=StatementParserWithoutClosing)
        important_note = get_important_note(not_included_statements="CLOSING")

    if "OPENING" not in total_type_considered and "CLOSING_OUTCOME" not in total_type_considered:
        parser = PydanticOutputParser(pydantic_object=StatementParserWithoutOpeningAndClosing)
        important_note = get_important_note(not_included_statements="EXTREME_END")

    return important_note, parser

class ClassificationView(APIView):
    def post(self, request):
        transcript_id = request.data['transcript_id']
        print(transcript_id)
        transcript = Transcription.objects.get(id=transcript_id)

        total_segments = len(transcript.segments)

        for index, segment in enumerate(transcript.segments):

            # INITIAL_PROCESSING
            # Get all the statement category types
            statement_type_considered, different_statement_definition = get_statements_data(index+1, total_segments)
            important_note, parser = get_important_note_and_parser(statement_type_considered, index + 1, total_segments)

            prompt_template = (
                "READ THE " + str(index+1) + " PART OF THE TRANSCRIPT (WHOLE TRANSCRIPT HAS TOTAL " + str(total_segments)
                +" PARTS, with overlaps of 200 words in each parts). \n"
                "The conversation is between a representative (REP) and a healthcare professional (HCP). "
                "\nRead the transcript's part line  by line: \n###\n{transcript}\n### \n\n"
                "Objective: Review and classify dialogue snippets from interactions between pharmaceutical sales representatives (REPs) and healthcare professionals (HCPs)."
                 " Each dialogue snippet may align with more than one of the following categories based on its content and intention:\n"
                 + ", ".join(statement_type_considered) + "\n"
                 " This requires discerning the REP's strategic approach towards initiating the conversation,"
                 " engaging in inquiry, presenting information, and steering towards a productive conclusion."
                 "\n\nDetailed Criteria for Classification::\n"
                 "\n{different_statement_definition}\n\n"
                 "Important Notes:\n"
                 "{important_note}\n\n"
                 "Maintain the original transcript form for authenticity, "
                 "focusing on the accuracy and relevance of each classification.\n\n"
                 "Assignment Execution: Thoroughly proceed through the dialogue transcripts,"
                 " identifying and classifying each REP statement according to the refined categories "
                 "and guidelines provided. Utilize a nuanced approach to determine the strategic intent "
                 "and outcome of the dialogue segments within the structured interaction framework."
                 "Instruction for your output format:\n"
                 "\n{format_instructions}\n\n"
                 "Output: "
            )

            prompt = PromptTemplate(
                input_variables=[
                    'transcript',
                    'different_statement_definition',
                    'important_note'
                ],
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            # llm = AzureOpenAI(azure_deployment='test', temperature=0, max_tokens=4000)
            # model_name = 'test'

            llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.2, max_tokens=4000)
            model_name = 'test3'

            classification_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

            response = classification_chain.run(
                transcript=segment,
                different_statement_definition=different_statement_definition,
                important_note=important_note
            )
            print("response before")
            print(response)
            print("response next")
            if "```" in response:
               response = re.findall(r'```(.*?)```', response, re.DOTALL)
            statements = eval(response)
            for key in statements.keys():
                if 'opening' in key:
                    category = 'OPENING'
                elif 'questioning' in key:
                    category = 'QUESTIONING'
                elif 'presenting' in key:
                    category = 'PRESENTING'
                elif 'closing_outcome_sentences' in key:
                    category = 'CLOSING_OUTCOME'

                print("Creating objects for : ", category)
                for statement in statements[key]:
                    classifcation_obj = StatementClassification(
                        category=category,
                        transcription=transcript,
                        statement=str(statement),
                        segment_number=index+1
                    )
                    classifcation_obj.save()
                print("Finished creating the Classification objects for the statement type: ", category)
        return Response({"message": "action is finished"}, status=status.HTTP_201_CREATED)

class TranscriptionView(APIView):

    def post(self, request):
        # Desired segment length and overlap
        segment_length = 4000  # Adjusted due to example length; use 1200 for your full text
        overlap = 100
        trans_obj = request.data
        # Create overlapping segments
        print(trans_obj)
        trans_obj['segments'] = create_overlapping_segments(trans_obj['text'], segment_length, overlap)

        serializer = TranscriptionSerializer(data=trans_obj)

        if serializer.is_valid():
            instance = serializer.save()
            instance.segments = create_overlapping_segments(trans_obj['text'], segment_length, overlap)
            instance.save()
            return Response({"transcript_id": instance.id, "number_of_segments": len(instance.segments)}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VectorDataView(APIView):

    def get(self, request):
        pass

    def post(self, request):
        statements = request.data['statements']
        collection_name = request.data['collection_name']
        storeData(collection_name, statements)
        return Response(
            {"message": "data has been stored in {collection_name}".format(collection_name=collection_name)},
            status=status.HTTP_201_CREATED
        )

class LevellingDataView(APIView):

    def post(self, request):
        transcript_id = request.data['transcript_id']
        print(transcript_id)
        transcript = Transcription.objects.get(id=transcript_id)
        sentences = transcript.statementclassification_set.all()
        category_map = {
            "OPENING": "OPENING",
            "QUESTIONING": "QUESTIONING",
            "PRESENTING": "PRESENTING",
            "CLOSING": "CLOSING_OUTCOME",
            "OUTCOME": "CLOSING_OUTCOME",
        }
        for category in ['OPENING', 'QUESTIONING', 'PRESENTING', 'CLOSING', 'OUTCOME']:
            statements = sentences.filter(category=category_map[category]).values('id', 'statement')
            classification_statements = sentences.filter(category=category_map[category])
            # statements = sentences.filter(category=category).filter(level=0).values('id', 'statement')
            # print(statements)

            statement_level_obj = StatementLevelPrompt.objects.filter(active=True).filter(category=category).first()
            if len(statements) == 0 or not statement_level_obj:
                continue
            prompt_template = (
                category + " statements: \n{statements}\n\n"
                "Objective: {objective}\n\n"
                "Evaluation Criteria:\n {evaluation_criteria}\n\n"
                "Score assignment criteria:\n {score_assignment_criteria}\n\n"
                "Instruction:\n {instruction}\n\n"
                "Examples:\n {examples}\n\n"
                "notes:\n {notes}\n\n"
                "Instruction for your output format:\n"
                 "\n{format_instructions}\n\n"
                 "Output: "
            )
            parser = PydanticOutputParser(pydantic_object=EvaluationResult)
            prompt = PromptTemplate(
                input_variables=[
                    'statements',
                    'objective',
                    'evaluation_criteria',
                    'score_assignment_criteria',
                    'instruction',
                    'examples',
                    'notes'
                ],
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            # llm = AzureOpenAI(azure_deployment='test', temperature=0, max_tokens=4000)
            # model_name = 'test'

            llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.2, max_tokens=4000)
            model_name = 'test3'
            rating_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
            response = rating_chain.run(
                statements=statements,
                objective=statement_level_obj.objective,
                evaluation_criteria=statement_level_obj.evaluation_criteria,
                score_assignment_criteria=statement_level_obj.score_assignment_criteria,
                instruction=statement_level_obj.instruction,
                examples=statement_level_obj.examples,
                notes=statement_level_obj.notes
            )
            print(response)
            print("response next")
            if "```" in response:
                response = re.findall(r'```(.*?)```', response, re.DOTALL)
            scored_statements = eval(response)
            for scored_statement in scored_statements["statements"]:
                print(scored_statement)
                statement_obj = StatementClassification.objects.get(id=scored_statement['id'])
                final_statement = FinalStatementWithLevel(
                    transcription=statement_obj.transcription,
                    category=category,
                    statement=statement_obj.statement,
                    level=scored_statement['level'],
                    confidence_score=scored_statement['confidence_score'],
                    reason_for_level=scored_statement['reason']
                )
                final_statement.save()

                print(final_statement)
            if category != "CLOSING":
                classification_statements.update(levelDone=True)
        return Response(
            {"message": "done" },
            status=status.HTTP_200_OK
        )

