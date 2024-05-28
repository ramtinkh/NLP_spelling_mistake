from opsdroid.skill import Skill
from opsdroid.matchers import match_regex, match_parse

class HelloSkill(Skill):
    @match_parse(r'repeat {speech}')
    async def hello(self, message):
        await message.respond(message.entities['speech']['value'])