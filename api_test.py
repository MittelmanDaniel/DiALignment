import asyncio
import streamlit as st
import aiohttp

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

if st.button("Fetch Data"):
    result = asyncio.run(fetch_data("https://dc2f-143-215-61-120.ngrok-free.app/test"))
    st.write(result)