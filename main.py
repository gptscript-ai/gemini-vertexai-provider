import json
import os
from typing import AsyncIterable, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from vertexai.preview.generative_models import Content, FunctionDeclaration, GenerationConfig, GenerationResponse, \
    GenerativeModel, Part, Tool

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"
app = FastAPI()
router = APIRouter()


def log(*args):
    if debug:
        print(*args)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/")
async def get_root():
    return 'ok'


@app.get("/v1/models")
def list_models() -> JSONResponse:
    content = {
        "data": [
            {
                "id": "gemini-1.0-pro",
                "name": "Gemini 1.0 Pro",
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
            }
        ]
    }
    return JSONResponse(content=content)


async def map_tools(req_tools: list | None = None) -> list[Tool] | None:
    tools: list[Tool] = []
    if req_tools is not None:
        for tool in req_tools:
            # TODO: determine if this mapping is required or not?
            # tool["function"]["parameters"]["type_"] = tool["function"]["parameters"].pop("type")
            # tool["function"]["parameters"]["type_"] = tool["function"]["parameters"]["type_"].upper()
            #
            # for prop in tool["function"]["parameters"]["properties"]:
            #     tool["function"]["parameters"]["properties"][prop]["type_"] = \
            #         tool["function"]["parameters"]["properties"][prop].pop("type")
            #
            #     tool["function"]["parameters"]["properties"][prop]["type_"] = \
            #         tool["function"]["parameters"]["properties"][prop]["type_"].uper()
            #     tool["function"]["parameters"]["properties"][prop].pop("description")

            convert_tool = Tool.from_function_declarations(
                [
                    FunctionDeclaration(
                        name=tool["function"]["name"],
                        parameters=tool["function"]["parameters"],
                        description=tool["function"]["description"]
                    )
                ]
            )
            tools.append(convert_tool)
        return tools
    return None


def merge_consecutive_dicts_with_same_value(list_of_dicts, key) -> list[dict]:
    merged_list = []
    index = 0
    while index < len(list_of_dicts):
        current_dict = list_of_dicts[index]
        value_to_match = current_dict.get(key)
        compared_index = index + 1
        while compared_index < len(list_of_dicts) and list_of_dicts[compared_index].get(key) == value_to_match:
            log("CURRENT DICT: ", current_dict)
            log("COMPARED DICT: ", list_of_dicts[compared_index])
            list_of_dicts[compared_index]["content"] = current_dict["content"] + "\n" + list_of_dicts[compared_index][
                "content"]
            current_dict.update(list_of_dicts[compared_index])
            compared_index += 1
        merged_list.append(current_dict)
        index = compared_index
    return merged_list


async def map_messages(req_messages: list) -> list[Content] | None:
    messages: list[Content] = []
    log(req_messages)

    if req_messages is not None:
        for message in req_messages:
            match message["role"]:
                case "system":
                    message['role'] = "user"
                case "user":
                    message['role'] = "user"
                case "assistant":
                    message['role'] = "model"
                case "model":
                    message['role'] = "model"
                case "tool":
                    message['role'] = "function"
                case _:
                    message['role'] = "user"
        req_messages = merge_consecutive_dicts_with_same_value(req_messages, "role")

        for message in req_messages:
            if 'tool_call_id' in message.keys():
                convert_message = Content(
                    role=message['role'],
                    parts=[Part.from_function_response(
                        name='test',
                        response={
                            'name': message['name'],
                            'content': message['content']
                        },
                    )]
                )
            elif 'tool_calls' in message.keys():
                tool_call_parts: list[Part] = []
                for tool_call in message['tool_calls']:
                    function_call = {
                        "functionCall": {
                            "name": tool_call['function']['name'],
                            "args": json.loads(tool_call['function']['arguments'])
                        }
                    }
                    tool_call_parts.append(Part.from_dict(function_call))
                convert_message = Content(
                    role=message['role'],
                    parts=tool_call_parts
                )
            elif 'content' in message.keys():
                convert_message = Content(
                    role=message['role'],
                    parts=[Part.from_text(message["content"])]
                )

            messages.append(convert_message)
        return messages
    return None


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    data = await request.body()
    data = json.loads(data)

    req_tools = data.get("tools", None)
    tools: list[FunctionDeclaration] | None = None
    tools = await map_tools(req_tools)

    # TODO: convert messages to gemini messages
    req_messages = data["messages"]
    messages = await map_messages(req_messages)

    temperature = data.get("temperature", None)
    if temperature is not None:
        temperature = float(temperature)

    stream = data.get("stream", False)

    top_k = data.get("top_k", None)
    if top_k is not None:
        top_k = float(top_k)

    top_p = data.get("top_p", None)
    if top_p is not None:
        top_p = float(top_p)

    max_output_tokens = data.get("max_tokens", None)
    if max_output_tokens is not None:
        max_output_tokens = float(max_output_tokens)

    model = GenerativeModel(data["model"])
    try:
        response = model.generate_content(contents=messages,
                                          tools=tools,
                                          stream=stream,
                                          generation_config=GenerationConfig(
                                              temperature=temperature,
                                              top_k=top_k,
                                              top_p=top_p,
                                              max_output_tokens=max_output_tokens,
                                          )
                                          )
        if not stream:
            return JSONResponse(content=jsonable_encoder(response.to_dict()))
        return StreamingResponse(async_chunk(response), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def async_chunk(chunks: Iterable[GenerationResponse]) -> \
        AsyncIterable[str]:
    for chunk in chunks:
        chunk = map_streaming_resp(chunk)
        log("MAPPED CHUNK: ", chunk.json())
        yield "data: " + chunk.json() + "\n\n"


def map_streaming_resp(chunk: GenerationResponse) -> ChatCompletionChunk:
    print("CHUNK: ", chunk)
    tool_calls = []
    if chunk.candidates[0].function_calls is not None:
        for idx, call in enumerate(chunk.candidates[0].function_calls):
            tool_calls.append({
                "index": idx,
                # TODO: is this required?
                "id": call.name + "_" + str(idx),
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(dict(call.args))
                },
                "type": "function"
            })
    content = None

    try:
        content = chunk.candidates[0].content.text
    except:
        content = None

    match chunk.candidates[0].content.role:
        case "system":
            role = "user"
        case "user":
            role = "user"
        case "assistant":
            role = "model"
        case "model":
            role = "assistant"
        case "function":
            role = "tool"
        case _:
            role = "user"

    try:
        finish_reason = map_finish_reason(str(chunk.candidates[0].finish_reason))
    except KeyError:
        finish_reason = None

    log("FINISH_REASON: ", finish_reason)

    resp = ChatCompletionChunk(
        id="0",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=content,
                    tool_calls=tool_calls,
                    role=role
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        created=0,
        model="",
        object="chat.completion.chunk",
    )
    return resp


def map_finish_reason(finish_reason: str) -> str:
    # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    if (finish_reason == "ERROR"):
        return "stop"
    elif (finish_reason == "FINISH_REASON_UNSPECIFIED" or finish_reason == "STOP"):
        return "stop"
    elif finish_reason == "SAFETY":
        return "content_filter"
    elif finish_reason == "STOP":
        return "stop"
    elif finish_reason == "1":
        return "stop"
    elif finish_reason == "0":
        return "stop"
    return finish_reason


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
