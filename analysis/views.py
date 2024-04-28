import re

from typing import List, Dict
from dotenv import load_dotenv

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from analysis.aiService.constants import ADDITIONAL_INFORMATION_FOR_CLASSIFICATION
from analysis.aiService.weaviateDb import fetch_top_k_content, storeData
from analysis.serializers import TranscriptionSerializer
from analysis.models import Transcription, Classification, StatementType, llmModel
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
        description="list of dictionaries containing closing and outcome statements")

def get_statements_data(part, total_parts):
    statement_types = StatementType.objects.filter(active=True)
    prompt_template = (" category: {category} , where "
              "the definition of the {category} category: ###{definition}### \n"
              # "the following are CORRECT sample statements of {category} : \n"
              # "###{valid_example}###\n\n"
              #  "the following are INCORRECT sample statements of {category} : \n"
              #  "###{invalid_example}###\n\n"
               "the instruction to do: ###{instruction}.### \n"
              )
    prompt = PromptTemplate(
        input_variables=[
            'category',
            'definition',
            'instruction',
            # 'valid_example',
            # 'invalid_example'
        ],
        template=prompt_template,
    )
    final_prompt  = ""
    for statement_type in statement_types:
        if(total_parts > 3 and (((part+1) * 1.0) / total_parts > .3)) and (statement_type.category == "OPENING"):
            continue
        if(total_parts > 3 and (((part+1) * 1.0) / total_parts < .7)) and (statement_type.category == "CLOSING_OUTCOME"):
            continue

        print(statement_type.category)
        final_prompt += "\n\n\n-----------for category "+statement_type.category+"statements -----------\n\n"
        # valid_example_db = fetch_top_k_content(str(statement_type.category) + '_CORRECT_EXAMPLE', k_size=15)
        # valid_example = [item['content'] for item in valid_example_db]
        # invalid_example_db = fetch_top_k_content(str(statement_type.category) + '_INCORRECT_EXAMPLE', k_size=15)
        # invalid_example = [item['content'] for item in invalid_example_db]
        prompt_value = prompt.format_prompt(
            category=statement_type.category,
            definition=statement_type.definition,
            instruction=statement_type.instruction,
            # valid_example='.\n'.join(valid_example),
            # invalid_example='.\n'.join(invalid_example)
        )
        final_prompt += str(prompt_value)

    return final_prompt

class ClassificationView(APIView):
    def post(self, request):
        transcript_id = request.data['transcript_id']
        print(transcript_id)
        transcript = Transcription.objects.get(id=transcript_id)
        parser = PydanticOutputParser(pydantic_object=StatementParser)

        total_segments = len(transcript.segments)

        for index, segment in enumerate(transcript.segments):
            prompt_template = (
                "READ THE " + str(index + 1) + " PART OF THE TRANSCRIPT (WHOLE TRANSCRIPT HAS TOTAL " + str(total_segments)
                +" PARTS, with overlaps of 200 words in each parts). \n"
                  "The conversation is between a representative (REP) and a healthcare professional (HCP). "
                "\nRead the transcript's part line  by line: \n###\n{transcript}\n### \n\n"
                "While classification of statement, Keep in mind the following things: \n"
                "\n " + str(ADDITIONAL_INFORMATION_FOR_CLASSIFICATION) + "\n\n"
                "{different_statement_data}\n\n Instruction for your output format:\n"
                 "\n{format_instructions}\n\n"
                 "Output: "
            )

            prompt = PromptTemplate(
                input_variables=[
                    'transcript',
                    'different_statement_data'],
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            # llm = AzureOpenAI(azure_deployment='test', temperature=0, max_tokens=4000)
            # model_name = 'test'

            llm = AzureChatOpenAI(azure_deployment='test2', temperature=0, max_tokens=4000)
            model_name = 'test2'

            classification_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

            different_statement_data = get_statements_data(index, total_segments)

            response = classification_chain.run(
                transcript=segment,
                different_statement_data=different_statement_data
            )
            print("response before")
            print(response)
            print("response next")
            if "```" in response:
               response = re.findall(r'```(.*?)```', response, re.DOTALL)
            statements = eval(response)
            single_statement = True
            for key in statements.keys():
                if 'opening' in key:
                    category = 'OPENING'
                elif 'questioning' in key:
                    category = 'QUESTIONING'
                elif 'presenting' in key:
                    category = 'PRESENTING'
                elif 'closing_outcome_sentences' in key:
                    category = 'CLOSING_OUTCOME'
                    single_statement = False
                if single_statement:
                    print("Creating objects for : ", category)
                    for statement in statements[key]:
                        classifcation_obj = Classification(
                            category=category,
                            transcription=transcript,
                            statement=statement,
                            level=0,
                            model_llm=model_name,
                            segment_number=index+1
                        )
                        classifcation_obj.save()
                    print("Finished creating the Classification objects for the statement type: ", category)
                else:
                    print("Creating objects for : ", category)
                    for statement in statements[key]:
                        classifcation_obj = Classification(
                            category=category,
                            transcription=transcript,
                            statement=str(statement),
                            level=0,
                            model_llm=model_name,
                            segment_number=index+1
                        )
                        classifcation_obj.save()
                    print("Finished creating the Classification objects for the statement type: ", category)
        return Response({"message": "action is finished"}, status=status.HTTP_201_CREATED)

class TranscriptionView(APIView):

    def post(self, request):
        # Desired segment length and overlap
        segment_length = 1200  # Adjusted due to example length; use 1200 for your full text
        overlap = 200
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
