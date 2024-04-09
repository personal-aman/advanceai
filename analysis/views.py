from typing import List

from .serializers import TranscriptionSerializer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Transcription, Classification, StatementType
from .aiService.watsonX import mixtralLLM

from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Create your views here.

class StatementParser(BaseModel):
    statements: List[str] = Field(description="list of different statements")

class ClassificationView(APIView):
    def post(self, request):
        transcript_id = request.data['transcript_id']
        print(transcript_id)
        transcript = Transcription.objects.get(id=transcript_id)
        statement_types = StatementType.objects.all()
        prompt_template = ("Read the transcript between a representative (REP) and a healthcare professional (HCP)."
                           "\n the transcript: ###{transcript}### \n\n"
                           "the instruction for you to do: {instruction} \n"
                           "You have to classify the statement according the category: {category} \n"
                           "the definition of the {category} category: {definition} \n"
                           "\n{format_instructions}\n"
                           "Output: ")
        parser = PydanticOutputParser(pydantic_object=StatementParser)
        prompt = PromptTemplate(
            input_variables=['transcript','category','definition', 'instruction'],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        classification_chain = LLMChain(llm=mixtralLLM, prompt=prompt)
        for statement_type in statement_types:
            print("running chain for the statement type: ", statement_type.category)
            response = classification_chain.run(
                transcript=transcript.text,
                category=statement_type.category,
                definition=statement_type.definition,
                instruction=statement_type.instruction
            )
            print("response before")
            print(response)
            print("response next")

            statements = eval(response.strip().replace("```", ""))
            for statement in statements["statements"]:
                classifcation_obj = Classification(
                    category=statement_type.category,
                    transcription=transcript,
                    statement=statement,
                    level=0
                )
                classifcation_obj.save()
            print("Finished creating the Classification objects for the statement type: ", statement_type.category)
        return Response({"message": "action is finished"}, status=status.HTTP_201_CREATED)

class TranscriptionView(APIView):

    def post(self, request):
        serializer = TranscriptionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)