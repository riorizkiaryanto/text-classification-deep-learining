SERVER_URL = "http://127.0.0.1:5000"

document.getElementById('text-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const text_input = document.getElementById('text').value;

    try {
        const response = await fetch(`${SERVER_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({text: text_input})
        });

        const data = await response.json();
        
        document.getElementById('prediction').innerText = `Prediction: ${data.category}, Sub-Category: ${data.subcategory}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('prediction').innerText = 'Error occurred while making prediction';
    }
});