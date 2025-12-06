from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class GetWeatherArgs(BaseModel):
    city: str = Field(..., description="Name of the city")
    unit: Literal["C", "F"] = Field(
        "C", description="Temperature unit: C for Celsius, F for Fahrenheit"
    )


if __name__ == "__main__":
    tool = Tool(
        function=ToolFunction(
            name="get_weather",
            description="Get weather data for a city",
            parameters=GetWeatherArgs.model_json_schema(),
        )
    )
    print(tool.model_dump())
