// Google Apps Script backing DATA_ENDPOINT in screen.html.
//
// This file is a reference copy only — Apps Script projects aren't stored in
// this git repo, they live in the Google account of whoever created the
// deployment. Paste this into Extensions -> Apps Script from inside the
// target Sheet (currently:
// https://docs.google.com/spreadsheets/d/11fB30GMa8GNcLraN_7CH8JRy2ECEwlM2joyiXv023Ks/edit)
// so the script is container-bound to it.
//
// screen.html POSTs { kind: 'trials' | 'surveys', headers, rows } with
// mode: 'no-cors' (fire-and-forget — the client never reads the response,
// so verify this works by checking the Sheet after a real playthrough, not
// by watching devtools).
//
// Deploying an update: Deploy -> Manage deployments -> edit the existing
// deployment -> "New version". Do NOT create a brand-new deployment for an
// update — that mints a new /exec URL and orphans the one already baked
// into screen.html's DATA_ENDPOINT constant.
//
// Deployment settings required: Execute as "Me", Who has access "Anyone" —
// players hitting this from a public build (itch.io, GitHub Pages, etc.)
// are not logged into the owning Google account.

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const { kind, headers, rows } = JSON.parse(e.postData.contents);
    if (!rows || !rows.length) return ContentService.createTextOutput('ok');

    const sheetName = kind === 'surveys' ? 'surveys' : 'trials';
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getSheetByName(sheetName) || ss.insertSheet(sheetName);
    if (sheet.getLastRow() === 0) sheet.appendRow(headers);
    sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows);
  } finally {
    lock.releaseLock();
  }
  return ContentService.createTextOutput('ok');
}
