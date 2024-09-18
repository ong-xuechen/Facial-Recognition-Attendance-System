/* admin base javascript */

/* admin navbar js scripts (user profile dropdown) */
$("#sidebarCollapse").click(function () {
    $(".sidebar").toggleClass("active");
});

$('.dropdown-toggle').dropdown();

/* admin enroll success page js script */
function openPopup() {
    document.getElementById("popupBox").style.display = "block";
}

function closePopup() {
    document.getElementById("popupBox").style.display = "none";
}

function sendNotification() {
    alert("Notification sent!");
    // Code to send the notification to the user side using appropriate methods or APIs
}

// js script for face training page //
window.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const progressBar = document.getElementById('progress');
    const popup = document.getElementById('popup');
    const popupText = document.getElementById('popup-text');
    const closeBtn = document.getElementById('close-btn');
    const okBtn = document.getElementById('ok-btn');

    startBtn.addEventListener('click', () => {
        startBtn.disabled = true;
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
                popup.style.display = 'block';
            } else {
                width++;
                progressBar.style.width = `${width}%`;
            }
        }, 50);
    });

    function closePopup() {
        popup.style.display = 'none';
        startBtn.disabled = false;
        progressBar.style.width = '0%';
    }

    closeBtn.addEventListener('click', closePopup);
    okBtn.addEventListener('click', closePopup);
});

// js script for face processing page //
var initialTableHtml; // Variable to store the initial HTML of the table

// Function to generate the table
function generateTable() {
    var numRecordsInput = document.getElementById("num_records");
    var numRecords = parseInt(numRecordsInput.value);

    var userTable = document.getElementById("user-table");
    var tableRows = userTable.getElementsByTagName("tr");

    if (!initialTableHtml) {
        initialTableHtml = userTable.innerHTML; // Store the initial HTML of the table
    }

    // Start from index 1 to exclude the header row
    for (var i = 1; i < tableRows.length; i++) {
        if (i <= numRecords) {
            tableRows[i].style.display = "table-row";
        } else {
            tableRows[i].style.display = "none";
        }
    }
}

// Function to reset the table
function resetTable() {
    var userTable = document.getElementById("user-table");
    userTable.innerHTML = initialTableHtml; // Restore the initial HTML of the table
}

// Function to start the enrollment process
function startEnrollment() {
    var progressBar = document.getElementById("progress-bar-inner");
    progressBar.style.width = "100%";

    // Show the popup box after 0.5s delay
    setTimeout(function () {
        var popup = document.getElementById("popup");
        popup.style.display = "block";
    }, 500);
}

// Function to proceed to training
function proceedToTrain() {
    // Code to proceed to training
}

// JavaScript code to handle closing the popup box and resetting the progress bar
window.addEventListener('DOMContentLoaded', function() {
    var popupClose = document.getElementById("popup-close");
    var popupOk = document.getElementById("popup-ok");
    var popup = document.getElementById("popup");
    var progressBar = document.getElementById("progress-bar-inner");

    // Function to close the popup box and reset the progress bar
    function closePopup() {
        popup.style.display = "none";
        progressBar.style.width = "0";
    }

    // Attach event listener to the cross icon
    popupClose.addEventListener('click', closePopup);

    // Attach event listener to the OK button
    popupOk.addEventListener('click', closePopup);
});

// js script for admin guide //
// scroll to top function
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
}


// other js scripts //
