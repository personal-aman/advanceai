import re

from dotenv import load_dotenv

from django.http import JsonResponse
from django.db.models import Max
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from analysis.aiService.weaviateDb import storeData
from analysis.serializers import TranscriptionSerializer
from analysis.models import Transcription, StatementClassification, StatementLevelPrompt, FinalStatementWithLevel
from analysis.utils import create_overlapping_segments, get_statements_data, get_important_note_and_parser

from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .outputParser import EvaluationResult
from .synctask import sync_classify_segments
from .tasks import save_transcription, classify_segments

load_dotenv()

# set_debug(True)

class ClassificationView(APIView):
    def post(self, request):
        transcript_id = request.data['transcript_id']
        print(transcript_id)
        transcript = Transcription.objects.get(id=transcript_id)

        total_segments = len(transcript.segments)

        for index, segment in enumerate(transcript.segments):

            # INITIAL_PROCESSING
            # Get all the statement category types
            statement_type_considered, different_statement_definition = get_statements_data(index + 1, total_segments)
            important_note, parser = get_important_note_and_parser(statement_type_considered, index + 1, total_segments)

            prompt_template = (
                "READ THE " + str(index+1) + " PART OF THE TRANSCRIPT (WHOLE TRANSCRIPT HAS TOTAL " + str(total_segments)
                +" PARTS, with overlaps of 200 words in each parts). \n"
                "The conversation is between a representative (REP) and one or more healthcare professionals (HCPs). "
                "\nRead the transcript's part line  by line: \n###\n{transcript}\n### \n\n"
                "Objective: Review and classify dialogues from interactions between pharmaceutical sales representatives (REPs) and healthcare professionals (HCPs)."
                 " Each dialogue may belong to MORE THAN ONE OR NONE of the following categories( based on its content, intention and definition of the category):\n"
                 + ", ".join(statement_type_considered) + "\n"
                 " This requires discerning the REP's strategic approach towards initiating the conversation,"
                 " engaging in inquiry, presenting information, and steering towards a productive conclusion."
                 "\n\nDetailed Definition for Classification::\n"
                 "\n{different_statement_definition}\n\n"
                 "Important Notes:\n"
                 "###{important_note}\n\n"
                 "Maintain the original transcript form for authenticity, "
                 "focusing on the accuracy and relevance of each classification.###\n\n"
                 "Assignment Execution: ### Thoroughly proceed through the transcripts,"
                 " identifying and classifying each of REP dialogues according to the category's definition, category's examples and category's invalid examples."
                 "Also take care of the guidelines and notes provided. Utilize a nuanced approach to determine the strategic intent "
                 "and outcome of the dialogue within the structured interaction framework.###"
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

            llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.4, max_tokens=4000)
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

    def get(self, request):
        transcripts = Transcription.objects.filter(docker_feteched=False)
        return Response({"pending_transcript": [transcript.id for transcript in transcripts]}, status=status.HTTP_200_OK)

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

def transcription_status_update(request, transcript_id):
    obj = Transcription.objects.filter(id=transcript_id).first()
    obj.docker_feteched = True
    obj.save()
    return JsonResponse({"message": "done"}, safe=False)

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
            statements = list(sentences.filter(category=category_map[category]).filter(levelDone=False).values('id', 'statement'))
            classification_statements = sentences.filter(category=category_map[category]).filter(levelDone=False)
            # statements = sentences.filter(category=category).filter(level=0).values('id', 'statement')
            # print(statements)

            statement_level_obj = StatementLevelPrompt.objects.filter(active=True).filter(category=category).first()
            if len(statements) == 0 or not statement_level_obj:
                continue
            prompt_template = (
                category + " statements: \n{statements}\n\n"
                "Objective: !!!{objective}!!!\n\n"
                "Evaluation Criteria:\n ###{evaluation_criteria}###\n\n"
                "Score assignment criteria:\n ###{score_assignment_criteria}###\n\n"
                "Instruction:\n ###{instruction}###\n\n"
                "Examples:\n ###{examples}###\n\n"
                "notes:\n ###{notes}###\n\n"
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

            llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.4, max_tokens=4000)
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





def highest_level_statements(request, transcript_id):
    categories = ['OPENING', 'QUESTIONING', 'PRESENTING', 'CLOSING', 'OUTCOME']
    results = []

    for category in categories:
        # Find the highest level statement in each category
        highest_level = FinalStatementWithLevel.objects.filter(transcription__id=transcript_id, category=category).aggregate(Max('level'))['level__max']

        if highest_level is not None:
            statement = FinalStatementWithLevel.objects.filter(transcription__id=transcript_id, category=category, level=highest_level).first()
            results.append({
                'statement': statement.statement,
                'level': statement.level,
                'category': statement.category,
                'reason_for_level': statement.reason_for_level,
                'confidence_score': statement.confidence_score
            })

    return JsonResponse(results, safe=False)

def highest_level_statements_docker_results(request, transcript_id):
    return JsonResponse(getDockerOutput(transcript_id), safe=False)


class FullProcessView(APIView):
    def post(self, request):
        segment_length = 6000  # Adjust as needed
        overlap = 100  # Adjust as needed
        print(type(request.data))
        serializer = TranscriptionSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            instance.segments = create_overlapping_segments(request.data['text'], segment_length, overlap)
            instance.save()
            classify_segments.delay(instance.id)
            return Response({"transcript_id": instance.id, "number_of_segments": len(instance.segments)}, status=status.HTTP_202_ACCEPTED)
        else:
            return JsonResponse(serializer.errors, safe=False, status=status.HTTP_400_BAD_REQUEST)

def getDockerOutput(transcript_id):
    categories = ['OPENING', 'QUESTIONING', 'PRESENTING', 'CLOSING', 'OUTCOME']
    category_map = {
        "OPENING": "Opening",
        "QUESTIONING": "Questioning",
        "PRESENTING": "Presenting",
        "CLOSING": "Closing",
        "OUTCOME": "Outcome",
    }
    results = []
    consolidated_score = []

    for category in categories:
        # Find the highest level statement in each category
        highest_level = \
        FinalStatementWithLevel.objects.filter(transcription__id=transcript_id, category=category).aggregate(
            Max('level'))['level__max']
        if highest_level is not None:
            statement = FinalStatementWithLevel.objects.filter(transcription__id=transcript_id, category=category,
                                                               level=highest_level).first()
            consolidated_score.append({
                "category": category_map[statement.category],
                "score": statement.level
            })

            all_statements = FinalStatementWithLevel.objects.filter(transcription__id=transcript_id, category=category).all()
            if category == 'CLOSING' or category == 'OUTCOME':
                sentences = [
                    {
                        # FIXME: change the values
                        "call_to_action": ["No CTA"],
                        "call_to_action_confidence": "0",
                        "criteria": str(sentence.level),
                        "category": category_map[sentence.category],
                        "criteria_confidence_score": sentence.confidence_score,
                        "sentence": sentence.statement,
                        "speaker": "REP"
                    }
                    for sentence in all_statements
                ]
            else:
                sentences = [
                    {
                        "criteria": str(sentence.level),
                        "criteria_confidence_score": sentence.confidence_score,
                        "sentence": sentence.statement,
                        "speaker": "REP"
                    }
                    for sentence in all_statements
                ]
            transcript_chunk = " ".join([sentence['sentence'] for sentence in sentences])
            if category == 'CLOSING':
                results.append({
                    "category": "Closing & Outcome",
                    "category_confidence_score": 1,
                    "consolidated_close_score": statement.level,
                    "sentences": sentences,
                    "transcript_chunks": transcript_chunk
                })
            elif category == 'OUTCOME':
                closing_results = results[-1]
                closing_results['consolidated_outcome_score'] = statement.level
                closing_results['sentences'] += sentences
                # results[-1] = closing_results
                print(closing_results)
                continue
            else:
                results.append({
                    "category": category_map[statement.category],
                    "category_confidence_score": 1,
                    "sentences": sentences,
                    "transcript_chunks": transcript_chunk
                })

    responseDict = {
        "consolidate_scores": consolidated_score,
        "transcript": results,
        "status": "Success",
        "code": "200",
        "error_msg": "None",
        "merged_speakers": [
            {"extra_speakers": {}},
            {"original_speaker": "2"},
            {"deleted": False},
            {"merged": False}
        ],
        "speaker_total_time": {
            "HCP_total_time": 50.32,
            "REP_total_time": 261.49
        },
        "consolidated_call_to_action_scores": [
            {
                "category": "Prescribe",
                "score": "0"
            },
            {
                "category": "Patient Id",
                "score": "0"
            },
            {
                "category": "MSL",
                "score": "0"
            },
            {
                "category": "Advocacy",
                "score": "0"
            },
            {
                "category": "Guidelines",
                "score": "0"
            },
            {
                "category": "Next Meeting",
                "score": "0"
            },
            {
                "category": "Send Info",
                "score": "0"
            },
            {
                "category": "No CTA",
                "score": "0"
            }
        ],
    }
    return responseDict

class FullSyncProcessView(APIView):
    def post(self, request):
        segment_length = 6000  # Adjust as needed
        overlap = 100  # Adjust as needed
        print(type(request.data))
        serializer = TranscriptionSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            instance.segments = create_overlapping_segments(request.data['text'], segment_length, overlap)
            instance.save()
            sync_classify_segments(instance.id)
            return Response(getDockerOutput(instance.id), status=status.HTTP_200_OK)
        else:
            return JsonResponse(serializer.errors, safe=False, status=status.HTTP_400_BAD_REQUEST)