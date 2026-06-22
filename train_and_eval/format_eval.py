import json
import common as c


class FormatExtractor:
    @staticmethod
    def qa_mc(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean BioQA and MedQA multiple choice task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = sample["question"] + "\n" + "\n".join(sample["options"])
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["answer"]
            out = inst + sample["answer"]
            return out
        elif finetune_format_official:
            question = sample["question"]
            max_choices = ["A", "B", "C", "D", "E"]
            choices = sample["choices"]
            n_choices = len(choices)
            answers = "\n".join(
                [
                    f"{choice}. {text}"
                    for choice, text in zip(max_choices[:n_choices], choices)
                ]
            )
            inst = question + "\n" + answers
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + max_choices[sample["answer"]]
            out = inst + max_choices[sample["answer"]]
            return out
        elif eval_format_official:
            question = sample["question"]
            max_choices = ["A", "B", "C", "D", "E"]
            choices = sample["choices"]
            n_choices = len(choices)
            answers = "\n".join(
                [
                    f"{choice}. {text}"
                    for choice, text in zip(max_choices[:n_choices], choices)
                ]
            )
            out = question + "\n" + answers
            return out
        else:
            # if any valid JSON is in output string, get it
            # if multiple valid JSONs exist, the first one is returned
            question, options, answer = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    qa_dict = json.loads(json_str)
                    question = qa_dict["question"].strip()
                    options = [opt.strip() for opt in qa_dict["options"]]
                    answer = qa_dict["answer"][0].strip()

                    if len(question) <= 15:
                        question = ""
                        raise QuestionLengthError(
                            "No question  found / below 15 characters."
                        )
                    elif len(options) != 4:
                        options = ""
                        raise OptionsNumberError(
                            "Number of answer options other than 4."
                        )
                    elif answer not in "ABCD":
                        answer = ""
                        raise AnswerFormatError(
                            "Answer does not include A, B, C, or D."
                        )
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([question, options, answer]), f"{error_msgs[-1]}"

            out_dict = {"question": question, "options": options, "answer": answer}
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def qa_yn(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean common sense QA yes-no task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = sample["question"] + "\n" + "\n".join(sample["options"])
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["answer"]
            out = inst + sample["answer"]
            return out
        elif finetune_format_official:
            letter_to_idx = {"yes": 0, "no": 1}
            question = sample["question"]
            max_choices = ["A", "B"]
            choices = ["yes", "no"]
            answers = "\n".join(
                [f"{choice}. {text}" for choice, text in zip(max_choices, choices)]
            )
            inst = question + "\n" + answers
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + max_choices[letter_to_idx[sample["answer"]]]
            out = inst + max_choices[letter_to_idx[sample["answer"]]]
            return out
        elif eval_format_official:
            letter_to_idx = {"yes": 0, "no": 1}
            question = sample["question"]
            max_choices = ["A", "B"]
            choices = ["yes", "no"]
            answers = "\n".join(
                [f"{choice}. {text}" for choice, text in zip(max_choices, choices)]
            )
            out = question + "\n" + answers
            return out
        else:
            # if any valid JSON is in output string, get it
            # if multiple valid JSONs exist, the first one is returned
            question, options, answer = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    qa_dict = json.loads(json_str)
                    if "question" in qa_dict:
                        question = qa_dict["question"].strip()
                    elif "statement" in qa_dict:
                        question = qa_dict["statement"].strip()
                    options = [opt.strip() for opt in qa_dict["options"]]
                    answer = qa_dict["answer"][0].strip()

                    if len(question) <= 15:
                        question = ""
                        raise QuestionLengthError(
                            "No question or statement found / shorter than 15 characters."
                        )
                    elif len(options) != 2:
                        options = ""
                        raise OptionsNumberError(
                            "Number of answer options other than 2."
                        )
                    elif answer not in "AB":
                        answer = ""
                        raise AnswerFormatError("Answer does not include A or B.")
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([question, options, answer]), f"{error_msgs[-1]}"

            out_dict = {"question": question, "options": options, "answer": answer}
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def recipe(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean recipe task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction = sample["Instruction"].strip()
            output_parts = sample["Output"].split("Steps:")
            out_dict = {
                "instruction": instruction,
                "ingredients": [
                    ing.strip() for ing in output_parts[0].split("\n")[1:] if ing
                ],
                "steps": [step.strip() for step in output_parts[1].split("\n") if step],
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = c.create_mistral_inst(sample["instruction"])
            ingredients = "\n".join(sample["ingredients"])
            steps = "\n".join(sample["steps"])
            # out = f"{inst}\nIngredients:\n{ingredients}\nSteps:\n{steps}"
            out = f"{inst}Ingredients:\n{ingredients}\nSteps:\n{steps}"
            return out
        elif finetune_format_official:
            inst = c.create_mistral_inst(sample["instruction"])
            ingredients = sample["ingredients"]
            steps = sample["steps"]
            # out = f"{inst}\nIngredients:\n{ingredients}\nSteps:\n{steps}"
            out = f"{inst}Ingredients:\n{ingredients}\nSteps:\n{steps}"
            return out
        elif eval_format_official:
            out = sample["instruction"]
            return out
        else:
            instruction, ingredients, steps = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    recipe_dict = json.loads(json_str)
                    instruction = recipe_dict["instruction"].strip()
                    ingredients = [
                        ing.strip() for ing in recipe_dict["ingredients"] if ing
                    ]
                    steps = [step.strip() for step in recipe_dict["steps"] if step]

                    if len(instruction) <= 15:
                        instruction = ""
                        raise QuestionLengthError(
                            "No instruction found / below 15 characters."
                        )
                    elif not type(ingredients) is list:
                        ingredients = ""
                        raise AnswerFormatError("Ingredients not in list format.")
                    elif len(ingredients) < 1:
                        ingredients = ""
                        raise AnswerFormatError("One or no ingredient found.")
                    elif not type(steps) is list:
                        steps = ""
                        raise AnswerFormatError("Steps not in list format.")
                    elif len(steps) < 1:
                        steps = ""
                        raise AnswerFormatError("One or no cooking step found.")
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([instruction, ingredients, steps]), f"{error_msgs[-1]}"

            out_dict = {
                "instruction": instruction,
                "ingredients": ingredients,
                "steps": steps,
            }
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def summarization(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean summarization task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = [
                part.strip() for part in sample["Instruction"].split("\n") if part
            ]
            out_dict = {
                "instruction": instruction_parts[0],
                "summary": sample["Output"].strip(),
                "long_but_clean_text": "\n".join(instruction_parts[1:]),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            inst = f"{sample['instruction']}\n{sample['long_but_clean_text']}"
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["summary"]
            out = inst + sample["summary"]
            return out
        elif finetune_format_official:
            inst = f"Please summarize the text below:\n{sample['article']}"
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["highlights"]
            out = inst + sample["highlights"]
            return out
        elif eval_format_official:
            out = f"Please summarize the text below:\n{sample['article']}"
            return out
        else:
            instruction, summary, long_but_clean_text = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    summary_dict = json.loads(json_str)
                    instruction = summary_dict["instruction"].strip()
                    summary = summary_dict["summary"].strip()
                    long_but_clean_text = summary_dict["long_but_clean_text"].strip()

                    if len(instruction) <= 15:
                        instruction = ""
                        raise QuestionLengthError(
                            "Instruction shorter than 15 characters."
                        )
                    elif len(summary) <= 100:
                        summary = ""
                        raise AnswerFormatError("Summary shorter than 100 characters.")
                    elif len(long_but_clean_text) <= 500:
                        long_but_clean_text = ""
                        raise AnswerFormatError(
                            "Long but clean text shorter than 500 characters."
                        )
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([instruction, summary, long_but_clean_text]), f"{error_msgs[-1]}"

            out_dict = {
                "instruction": instruction,
                "summary": summary,
                "long_but_clean_text": long_but_clean_text,
            }
        out_json = json.dumps(out_dict)

        return out_json


class QuestionLengthError(Exception):
    pass


class OptionsNumberError(Exception):
    pass


class AnswerFormatError(Exception):
    pass
