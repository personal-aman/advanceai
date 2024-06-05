from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from analysis.aiService.constants import get_important_note
from analysis.models import StatementClassificationTypePrompt
from analysis.outputParser import StatementParser, StatementParserWithoutOpening, StatementParserWithoutClosing, \
    StatementParserWithoutOpeningAndClosing


def create_overlapping_segments(text, seg_length, overlap):
    # Split the text into words
    words = text.split()

    # Initialize segments and start position
    segments = []
    start = 0

    # Loop through the text and create segments
    while start < len(words):
        # End position is start plus segment length
        end = start + seg_length

        # Append the segment from start to end
        segments.append(' '.join(words[start:end]))

        # Update start position with segment length minus overlap
        start += (seg_length - overlap)

    return segments


def get_statements_data(part, total_parts):
    statement_types = StatementClassificationTypePrompt.objects.filter(active=True).order_by("id")
    prompt_template = (" category: {category} , where "
              "the definition of the {category} category: ###{definition}### \n\n"
              "the examples of the {category} category: ###{examples}### \n\n"
              "the examples that doesn't belong to {category} category: ###{invalid_examples}### \n"
              )
    prompt = PromptTemplate(
        input_variables=[
            'category',
            'definition',
            'examples',
            'invalid_examples'
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
            examples=statement_type.examples,
            invalid_examples=statement_type.invalid_examples
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
