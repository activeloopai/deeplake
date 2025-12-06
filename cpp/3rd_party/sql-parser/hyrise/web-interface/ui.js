
function SetUiStateRunning() {
	$('#resultTable').html('');
	$('#resultInfo').html('waiting for result...');
	$('#msgContainer').attr('class', 'alert alert-warning');
	$('#performanceDataTable tbody').html('');
}

function SetUiStateError(msg) {
	$('#resultInfo').html(msg);
	$('#msgContainer').attr('class', 'alert alert-danger');
	$('#performanceDataTable tbody').html('');
}

function SetUiStateSuccess(msg) {
	$('#msgContainer').attr('class', 'alert alert-success');
	$('#resultInfo').html(msg);
}

function GetHyriseUrl() {
	var endpointUrl = $('#endpointInput').val();
	return endpointUrl;
}

function GetQuery() {
	var query = $('#queryInput').val();
	// Check whether a part of the query has been selected
	var selectedText = GetSelectedText();
	if (query.indexOf(selectedText) >= 0) {
		query = selectedText;
	}
	return query;
}


/**
 * Bootstrap
 */
$(function() {
	LoadSampleQueries('sample-queries.sql');

	// Simple query submit
	$('#submitBtn').click(function() {
		SetUiStateRunning();
		var query = GetQuery();
		var hyrise = new HyriseConnector(GetHyriseUrl());

		hyrise.executeSQL(query, function(result) {
			console.log("Query result: ", result);
			// On Success
			SetUiStateSuccess('Result contains ' + result.real_size + ' rows');
			UpdateResultTable(result);
			UpdatePerformanceData(result.performanceData);

		}, function(xhr, status, error) {
			// console.log(arguments);
			// On Error
			var msg = 'Error when fetching result. Possibly no connection to Hyrise.';
			if (xhr.responseJSON) msg = xhr.responseJSON.error[0];
			SetUiStateError(msg);
		});
	});

	$('#queryInput').keypress(function(evt) {
		if (evt.keyCode == 13 && evt.shiftKey) {
			$('#submitBtn').click();
			return false;
		}
		return true;
	});


	// Benchmark submit
	$('#benchmarkBtn').click(function() {
		SetUiStateRunning();
		var query = GetQuery();
		var hyrise = new HyriseConnector(GetHyriseUrl());

		// TODO: hardcoded 5
		var numRuns = parseInt($('#benchmarkInput').val());
		hyrise.benchmarkSQL(query, numRuns, function(result) {
			console.log("Benchmark result: ", result);
			UpdatePerformanceData(result.performanceData);
			SetUiStateSuccess('Success! See PerformanceData for results.');
		});
	});

	// Setup Table triggers
	var table = document.querySelector('#performanceDataTable');
	$('#performanceDataTable thead th').click(function() {
		var key = $(this).attr('data-key');

		if (table._sortKey && table._sortKey == key) table._asc = !table._asc;
		else table._asc = true;
		var sign = (table._asc) ? 1 : -1;

		if (table._data) {
			table._sortKey = key;
			SortTableData(table);
			InsertPerformanceData(table._data);
		}
	});
});


function GetSelectedText() {
    var text = "";
    if (window.getSelection) {
        text = window.getSelection().toString();
    } else if (document.selection && document.selection.type != "Control") {
        text = document.selection.createRange().text;
    }
    return (text === "") ? null : text;
}


function LoadSampleQueries(url) {
	$.get(url, function(data) {
		var lines = data.split('\n');
		var name, query = "", isBuggy = false;
		$.each(lines, function(i, line) {
			if (line[0] == '#') {
				// Append last query
				if (name && !isBuggy) AddSampleQuery(name, query);
				if (name && isBuggy)  AddBuggyQuery(name, query);

				// New query
				isBuggy = (line[1] == '!');
				name = line.substring((isBuggy) ? 3 : 2);
				query = "";
			} else {
				query += line + '\n';
			}
		});

		if (name && !isBuggy) AddSampleQuery(name, query);
		if (name && isBuggy)  AddBuggyQuery(name, query);
	});
}


function AddSampleQuery(name, query) {
	var btn = $('<button type="button" class="btn btn-sm btn-success">' + name + '</button>');
	btn.click(function(evt) {
		$('#queryInput').val(query);
		if (!evt.shiftKey) {
			$('#submitBtn').click();
		}
	});
	$('#sampleQueries').append(btn);
}

function AddBuggyQuery(name, query) {
	var btn = $('<button type="button" class="btn btn-sm btn-danger">' + name + '</button>');
	btn.click(function(evt) {
		$('#queryInput').val(query);
		if (!evt.shiftKey) {
			$('#submitBtn').click();
		}
	});
	$('#buggyQueries').append(btn);
}

function CreateElement(tag, value) {
	return $('<' + tag + '>' + value + '</' + tag + '>');
};

function UpdateResultTable(result) {
	// Present result json in result-view
	var table = $('#resultTable');
	table.html('');

	var th = $('<tr>');
	$.each(result.header, function(i, val) {
		th.append($('<th>' + val + '</th>'));
	});
	table.append(th);

	// Limit the rows to be displayed to 100
	for (var i = 0; i < 100 && i < result.real_size; ++i) {
		var row = result.rows[i];
		var tr = $('<tr>');
		$.each(row, function(j, val) {
			tr.append($('<td>' + val + '</td>'));
		});
		table.append(tr);
	}
};

function UpdatePerformanceData(performanceData) {
	var table = document.querySelector('#performanceDataTable');

	var timeStrings = [
		(performanceData.queryTaskTime + performanceData.parseTime).toFixed(3),
		performanceData.queryTaskTime.toFixed(3),
		performanceData.parseTime.toFixed(3),
		performanceData.totalTime.toFixed(3),
	]
	var timeInfo =  + ','
	$('#timeInfo').html(timeStrings.join(','));

	// Sort and insert into table
	var tableData = performanceData.operators;
	table._data = tableData;
	SortTableData(table);
	InsertPerformanceData(tableData);
};

function InsertPerformanceData(performanceData) {
	var tbody = $('#performanceDataTable tbody');
	tbody.html('');

	$.each(performanceData, function(i, data) {
		if (!data.time_ms) data.time_ms = data.endTime - data.startTime;

		var tr = $('<tr>');
		tr.append(CreateElement('td', data.id));
		tr.append(CreateElement('td', data.name));
		tr.append(CreateElement('td', data.duration));
		tr.append(CreateElement('td', data.time_ms.toFixed(6)));
		tr.append(CreateElement('td', data.startTime));
		tr.append(CreateElement('td', data.endTime));
		tbody.append(tr);
	});
}

function SortTableData(table) {
	if (!table._sortKey) table._sortKey = 'startTime';
	if (!('_asc') in table) table._asc = true;

	var key = table._sortKey;
	var sign = (table._asc) ? -1 : 1;
	table._data.sort(function(a, b) {
		if (a[key].localeCompare) return sign * a[key].localeCompare(b[key]);
		return sign * (a[key] - b[key]);
	});
}