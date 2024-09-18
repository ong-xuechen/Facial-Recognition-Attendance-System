/* face registration form javascript codes */
function openFileExplorer() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.png, .jpg, .jpeg, .gif, .jfif'; // Accepts only specified file formats
    fileInput.style.display = 'none';
    fileInput.multiple = true; // Enable multiple file selection

    fileInput.addEventListener('change', function (event) {
        const files = event.target.files;
        const supportedFormats = ['image/png', 'image/jpeg', 'image/gif', 'image/jfif'];

        Array.from(files).forEach(function (file) {
            if (supportedFormats.includes(file.type)) {
                displayImagePreview(file);
            } else {
                displayErrorMessage(`Unsupported file format: ${file.name}. Please select a PNG, JPG, JPEG, GIF, or JFIF file.`);
            }
        });
    });

    document.body.appendChild(fileInput);
    fileInput.click();
    document.body.removeChild(fileInput);
}

function displayImagePreview(file) {
    const uploadContainer = document.querySelector('.reg-upload-container');
    const imagePreview = uploadContainer.querySelector('#image-preview');
    const message = uploadContainer.querySelector('.message');

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = document.createElement('img');
        img.src = event.target.result;
        imagePreview.appendChild(img);
    };

    reader.readAsDataURL(file);

    const successMessage = document.createElement('div');
    successMessage.innerText = `File upload successful: ${file.name}`;
    successMessage.style.color = 'green';
    message.appendChild(successMessage);
}

function validateForm() {
    const formGroups = document.querySelectorAll('.reg-form-group');
    const errorMessages = [
        'Please enter your first name',
        'Please enter your last name',
        'Please enter your IDs',
        'Please enter a valid email address',
        'Please enter your designation'
    ];

    clearErrorMessages();

    const reversedFormGroups = Array.from(formGroups).reverse();
    reversedFormGroups.forEach(function (formGroup, index) {
        const input = formGroup.querySelector('input');
        const value = input.value.trim();
        if (value === '') {
            displayErrorMessage(errorMessages[index]);
            input.focus();
            return;
        }
    });

    const uploadContainer = document.querySelector('.reg-upload-container');
    if (uploadContainer.querySelectorAll('img').length === 0) {
        displayErrorMessage('Please upload at least one photo');
    }
}

function displayErrorMessage(message) {
    const uploadContainer = document.querySelector('.reg-upload-container');
    const errorMessage = document.createElement('div');
    errorMessage.innerText = message;
    errorMessage.style.color = 'red';
    uploadContainer.querySelector('.message').appendChild(errorMessage);
}

function clearErrorMessages() {
    const uploadContainer = document.querySelector('.reg-upload-container');
    const message = uploadContainer.querySelector('.message');
    message.innerHTML = '';
}

function cancelForm() {
    if (confirm('Are you sure you want to cancel?')) {
        const formGroups = document.querySelectorAll('.reg-form-group');
        const uploadContainer = document.querySelector('.reg-upload-container');

        // Clear input values
        formGroups.forEach(function (formGroup) {
            const input = formGroup.querySelector('input');
            input.value = '';
        });

        // Clear uploaded photos
        const imagePreview = uploadContainer.querySelector('#image-preview');
        imagePreview.innerHTML = '';

        // Clear error messages
        clearErrorMessages();
    }
}

function highlightUploadContainer(event) {
    event.preventDefault();
    const uploadContainer = document.querySelector('.reg-upload-container');
    uploadContainer.classList.add('highlight-dragover');
}

function unhighlightUploadContainer(event) {
    event.preventDefault();
    const uploadContainer = document.querySelector('.reg-upload-container');
    uploadContainer.classList.remove('highlight-dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    const uploadContainer = document.querySelector('.reg-upload-container');
    uploadContainer.classList.remove('highlight-dragover');

    const files = event.dataTransfer.files;
    const supportedFormats = ['image/png', 'image/jpeg', 'image/gif', 'image/jfif'];

    Array.from(files).forEach(function (file) {
        if (supportedFormats.includes(file.type)) {
            displayImagePreview(file);
        } else {
            displayErrorMessage(`Unsupported file format: ${file.name}. Please select a PNG, JPG, JPEG, GIF, or JFIF file.`);
        }
    });
}

// audio reader
const audioIcon = document.getElementById('audio-icon');
const regFormSubtext = document.querySelector('.reg-form-subtext');
let isSpeaking = false;
let speech = null;

audioIcon.addEventListener('click', function() {
    const text = regFormSubtext.innerText;

    if ('speechSynthesis' in window) {
        if (isSpeaking) {
            speechSynthesis.cancel();
            isSpeaking = false;
        } else {
            speech = new SpeechSynthesisUtterance(text);
            speech.lang = 'en-US';

            speechSynthesis.addEventListener('voiceschanged', function() {
                const voices = speechSynthesis.getVoices();
                const femaleVoice = voices.find(voice => voice.name === 'Google US English Female');
                speech.voice = femaleVoice;

                speechSynthesis.speak(speech);
                isSpeaking = true;
            });
        }
    } else {
        console.log('Speech synthesis not supported.');
    }
});

/* user guide js scripts (scroll-to-top button) */
// Scroll to top function
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* other js scripts (terms-and-conditions validation) */
function validateForm() {
    var checkBox = document.getElementById("agreeCheckbox");
    if (!checkBox.checked) {
        alert("You must agree to the terms and conditions.");
        return false;
    }
    return true;
}

/* js script for notification icon at navbar */
// Prevent dropdown menu from opening when clicking on the notification icon
document.getElementById('notificationBell').addEventListener('click', function (event) {
    event.stopPropagation();
});

// notification icon triggered by message sent from admin side