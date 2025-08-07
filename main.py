import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from manager import AdventureManager
from models import TripQuery


async def main() -> None:
    """
    Main entry point for the AdventureBot application.
    Creates a sample trip query and runs the adventure planning process.
    """
    # Sample trip query data
    query = TripQuery(
        start_date="2025-12-01",
        end_date="2025-12-14",
        location="Toronto",
        participant_number=2,
        participant_ages=[32, 35]
    )
    
    # Initialize and run the adventure manager
    await AdventureManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())