from email import message
import logging
from opsdroid.skill import Skill
from opsdroid.matchers import match_regex, match_parse
from opsdroid.matchers import match_event, match_crontab, match_parse, match_regex
from opsdroid.events import Message, JoinRoom, UserInvite, LeaveRoom

from transformers import AutoTokenizer, AutoConfig, AutoModelWithLMHead
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
import torch

model_path = "modelAli"
# load the model
# model = model.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    bos_token='<s>', 
    eos_token='</s>', 
    pad_token='<pad>',
    unk_token='<unk>'
)
tokenizer.add_special_tokens({
    "bos_token": '</s>',
    "eos_token": '</s>', 
    "pad_token": '<pad>',
    "unk_token": '<unk>'
})

class AcceptDirectMessageSkill(Skill):
    @match_event(UserInvite)
    async def accept_direct_message(self, event):
        logging.info(f"accept_direct_message -> {event}")
        await message.respond("You can enter any persian text. We will correct typo and different mistakes.\nEnter your input:")

    @match_event(JoinRoom)
    async def join_room(self, event):
        logging.info(f"new-joining -> {event}")
        await message.respond("Wellcome to the best bot ever.\nEnter your input to get the correct one:")

class SpellingSkill(Skill):
    @match_parse(r'{speech}')
    async def correction(self, message):
        correct_text = ""

        sample_input = message.entities['speech']['value']
        sample_input_ids = torch.tensor(tokenizer([sample_input])["input_ids"])
        sample_input_ids = sample_input_ids.to(device)

        sample_outputs = model.module.generate(
            input_ids=sample_input_ids,
            do_sample=True,
            top_k=50,
            max_length=50,
            top_p=0.95,
            num_return_sequences=1
        )
        for i, sample_output in enumerate(sample_outputs):
            correct_text = tokenizer.decode(sample_output, skip_special_tokens=False)
            correct_text = correct_text.replace("<|startoftext|>", "\n")
            correct_text = correct_text.replace("<s>", "")
            correct_text = correct_text.replace("</s>", "")
            correct_text = correct_text.replace("<sep>", "\n")

        await message.respond(correct_text)